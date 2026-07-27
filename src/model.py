# Model for Training & testing

from __future__ import division

import csv
import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch
from tqdm import tqdm

from src.archs import discriminator, generator, vgg_16
from PIL import Image
from utils.ops import l1_loss, content_loss, style_loss, angular_error


class Model(object):
    """
    Main model.
    @author: Zhe He
    @contact: zhehe@student.ethz.ch
    """
    def __init__(self, params):
        """init

        Parameters
        ----------
        params: dict.
        """

        self.params = params
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        (self.train_iter, self.valid_iter,
         self.test_iter, self.train_size) = self.data_loader()

        # building graph
        (self.x_r, self.angles_r, self.labels, self.x_t,
         self.angles_g, self.sides) = self.train_iter.get_next()

        (self.x_valid_r, self.angles_valid_r, self.labels_valid,
         self.x_valid_t, self.angles_valid_g, self.sides_valid) = self.valid_iter.get_next()

        (self.x_test_r, self.angles_test_r, self.labels_test,
         self.x_test_t, self.angles_test_g, self.sides_test) = self.test_iter.get_next()

        self.x_g = generator(self.x_r, self.angles_g)
        self.x_recon = generator(self.x_g, self.angles_r, reuse=True)

        self.angles_valid_g = tf.random_uniform(
            [params.batch_size, 2], minval=-1.0, maxval=1.0)

        self.x_valid_g = generator(self.x_valid_r, self.angles_valid_g,
                                   reuse=True)

        # reconstruction loss
        self.recon_loss = l1_loss(self.x_r, self.x_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss()

        # regression losses and adversarial losses
        (self.d_loss, self.g_loss, self.reg_d_loss,
         self.reg_g_loss, self.gp) = self.adv_loss()

        # update operations for generator and discriminator
        self.d_op, self.g_op = self.add_optimizer()

        # adding summaries
        self.summary = self.add_summary()

        # initialization operation
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

    @staticmethod
    def _image_stem(image_path):
        return os.path.splitext(os.path.basename(str(image_path)))[0]

    def _manifest_record(self, image_path, angles, label, side):
        subject_id = label
        if not isinstance(subject_id, str):
            subject_id = int(subject_id) + 1
        return {
            'subject_id': subject_id,
            'eye_type': side,
            'image_stem': self._image_stem(image_path),
            'image_path': image_path,
            'pitch': float(angles[1]),
            'yaw': float(angles[0]),
        }

    def _manifest_pairs_from_image_data(self, image_data):
        pairs = []
        for index in range(len(image_data.test_images)):
            source_row = self._manifest_record(
                image_data.test_images[index],
                image_data.test_angles_r[index],
                image_data.test_labels[index],
                image_data.test_sides[index])
            target_row = self._manifest_record(
                image_data.test_images_t[index],
                image_data.test_angles_g[index],
                image_data.test_labels[index],
                image_data.test_sides[index])
            pairs.append((source_row, target_row))
        return pairs

    def data_loader(self):
        """ load traing and testing dataset """

        hps = self.params

        dataset_name = getattr(hps, 'dataset', 'magia')
        if dataset_name == 'xgaze':
            from src.xgaze_data_loader import ImageData

            image_data_class = ImageData(
                load_size=hps.image_size,
                channels=3,
                data_path=hps.data_path,
                ids=hps.ids,
                metadata_path=getattr(hps, 'metadata_path', None),
                image_dir=getattr(hps, 'image_dir', None),
                max_train_samples=getattr(hps, 'max_train_samples', None),
                max_eval_samples=getattr(hps, 'max_eval_samples', 20),
                min_lightness=getattr(hps, 'min_lightness', None))
            image_data_class.preprocess()
            self.eval_manifest_pairs = image_data_class.test_pairs

            train_dataset = image_data_class.make_dataset(
                'train', hps.batch_size)
            valid_dataset = image_data_class.make_dataset(
                'val', hps.batch_size)
            test_dataset = image_data_class.make_dataset(
                'test', hps.batch_size)

            train_dataset_iterator = train_dataset.make_one_shot_iterator()
            valid_dataset = valid_dataset.make_one_shot_iterator()
            test_dataset_iterator = test_dataset.make_one_shot_iterator()

            return (train_dataset_iterator,
                    valid_dataset,
                    test_dataset_iterator,
                    image_data_class.train_size)

        if dataset_name in ['columbia', 'magia']:
            from src.columbia_data_loader import ImageData
        else:
            from src.data_loader import ImageData

        if dataset_name in ['columbia', 'magia']:
            image_data_class = ImageData(
                load_size=hps.image_size,
                channels=3,
                data_path=hps.data_path,
                ids=hps.ids,
                metadata_path=getattr(hps, 'metadata_path', None),
                root_path=getattr(hps, 'image_dir', None))
        else:
            image_data_class = ImageData(load_size=hps.image_size,
                                         channels=3,
                                         data_path=hps.data_path,
                                         ids=hps.ids)
        image_data_class.preprocess()
        if dataset_name in ['columbia', 'magia']:
            self.eval_manifest_pairs = self._manifest_pairs_from_image_data(
                image_data_class)
        else:
            self.eval_manifest_pairs = []

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.train_images,
             image_data_class.train_angles_r,
             image_data_class.train_labels,
             image_data_class.train_images_t,
             image_data_class.train_angles_g,
             image_data_class.train_sides))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.test_images,
             image_data_class.test_angles_r,
             image_data_class.test_labels,
             image_data_class.test_images_t,
             image_data_class.test_angles_g,
             image_data_class.test_sides))

        train_dataset = train_dataset.apply(
            shuffle_and_repeat(train_dataset_num)).apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        valid_dataset = test_dataset.apply(
            shuffle_and_repeat(test_dataset_num)).apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        test_dataset = test_dataset.apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        train_dataset_iterator = train_dataset.make_one_shot_iterator()
        valid_dataset = valid_dataset.make_one_shot_iterator()
        test_dataset_iterator = test_dataset.make_one_shot_iterator()

        return (train_dataset_iterator,
                valid_dataset,
                test_dataset_iterator,
                train_dataset_num)

    def adv_loss(self):
        """Build sub graph for discriminator and gaze estimator

        Returns
        -------
        d_loss: scalar, adversarial loss for training discriminator.
        g_loss: scalar, adcersarial loss ofr training generator.
        reg_loss_d: scalar, MSE loss for training gaze estimator
        reg_loss_g: scalar, MSE loss for training generator
        gp: scalar, gradient penalty
        """

        hps = self.params

        gan_real, reg_real = discriminator(hps, self.x_r)
        gan_fake, reg_fake = discriminator(hps, self.x_g, reuse=True)

        eps = tf.random_uniform(shape=[hps.batch_size, 1, 1, 1], minval=0.,
                                maxval=1.)
        interpolated = eps * self.x_r + (1. - eps) * self.x_g
        gan_inter, _ = discriminator(hps, interpolated, reuse=True)
        grad = tf.gradients(gan_inter, interpolated)[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(slopes - 1.))

        d_loss = (-tf.reduce_mean(gan_real) +
                  tf.reduce_mean(gan_fake) + 10. * gp)
        g_loss = -tf.reduce_mean(gan_fake)

        reg_loss_d = tf.losses.mean_squared_error(self.angles_r, reg_real)
        reg_loss_g = tf.losses.mean_squared_error(self.angles_g, reg_fake)

        return d_loss, g_loss, reg_loss_d, reg_loss_g, gp

    def feat_loss(self):
        """
        build the sub graph of perceptual matching network

        Returns
        -------
        c_loss: scalar, content loss
        s_loss: scalar, style loss
        """

        content_layers = ["vgg_16/conv5/conv5_3"]
        style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                        "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

        _, endpoints_mixed = vgg_16(
            tf.concat([self.x_g, self.x_t], 0))

        c_loss = content_loss(endpoints_mixed, content_layers)
        s_loss = style_loss(endpoints_mixed, style_layers)

        return c_loss, s_loss

    def optimizer(self, lr):
        """Return an optimizer

        Parameters
        ----------
        lr: learning rate.

        Returns
        -------
        tensorflow Optimizer instance.
        """

        hps = self.params

        if hps.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        if hps.optimizer == 'adam':
            return tf.train.AdamOptimizer(lr,
                                          beta1=hps.adam_beta1,
                                          beta2=hps.adam_beta2)
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):
        """Add an optimizer.

        Returns
        -------
        g_op: update operation for generator.
        d_op: update operation for discriminator.
        """

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        g_opt = self.optimizer(self.lr)
        d_opt = self.optimizer(self.lr)

        g_loss = (self.g_loss + 5.0 * self.reg_g_loss +
                  50.0 * self.recon_loss +
                  100.0 * self.s_loss + 100.0 * self.c_loss)
        d_loss = self.d_loss + 5.0 * self.reg_d_loss

        g_op = g_opt.minimize(loss=g_loss,
                              global_step=self.global_step,
                              var_list=g_vars)
        d_op = d_opt.minimize(loss=d_loss,
                              global_step=self.global_step,
                              var_list=d_vars)

        return d_op, g_op

    def add_summary(self):
        """Add summary operation.

        Return
        ------
        summary_op: tf summary.
        """

        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('reg_d_loss', self.reg_d_loss)
        tf.summary.scalar('reg_g_loss', self.reg_g_loss)
        tf.summary.scalar('gp', self.gp)
        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('c_loss', self.c_loss)
        tf.summary.scalar('s_loss', self.s_loss)

        tf.summary.image('real', (self.x_r + 1) / 2.0, max_outputs=5)
        tf.summary.image('fake', tf.clip_by_value(
            (self.x_g + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('recon', tf.clip_by_value(
            (self.x_recon + 1) / 2.0, 0., 1.), max_outputs=5)

        tf.summary.image('x_test', tf.clip_by_value(
            (self.x_valid_r + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('x_test_fake', tf.clip_by_value(
            (self.x_valid_g + 1) / 2.0, 0., 1.), max_outputs=5)

        summary_op = tf.summary.merge_all()

        return summary_op

    def train(self):
        """Train the model and save checkpoints.
        """

        hps = self.params

        num_epoch = hps.epochs
        train_size = self.train_size
        batch_size = hps.batch_size
        learning_rate = hps.lr

        num_iter = train_size // batch_size

        summary_dir = os.path.join(hps.log_dir, 'summary')
        model_path = os.path.join(hps.log_dir, 'model.ckpt')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:

            # init
            sess.run([self.init_op])

            summary_writer = tf.summary.FileWriter(summary_dir,
                                                   graph=sess.graph)

            saver = tf.train.Saver(max_to_keep=3)

            variables_to_restore = slim.get_variables_to_restore(
                include=['vgg_16'])
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, hps.vgg_path)

            try:

                for epoch in tqdm(range(num_epoch), desc="Epoch", position=0):

                    print("Epoch: %d" % epoch)

                    if epoch >= hps.epochs / 2:

                        learning_rate = (2. - 2. * epoch / hps.epochs) * hps.lr

                    pbar = tqdm(range(num_iter), desc="Iter", position=1, leave=False)
                    for it in pbar:

                        feed_d = {self.lr: learning_rate}

                        _, d_loss = sess.run([self.d_op, self.d_loss], feed_dict=feed_d)

                        if it % 5 == 0:
                            _, g_loss = sess.run([self.g_op, self.g_loss], feed_dict=feed_d)
                        
                        pbar.set_postfix(d_loss=d_loss, g_loss=g_loss)

                        if it % hps.summary_steps == 0:

                            summary, global_step = sess.run(
                                [self.summary, self.global_step],
                                feed_dict=feed_d)
                            summary_writer.add_summary(summary, global_step)
                            summary_writer.flush()
                            saver.save(sess, model_path,
                                       global_step=global_step)

            except KeyboardInterrupt:
                print("stop training")

    def eval(self):
        """ Evaluation. """
        hps = self.params

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)

        x_fake = generator(self.x_test_r, self.angles_test_g, reuse=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():
                saver.restore(test_sess, checkpoint)

                imgs_dir = os.path.join(hps.log_dir, 'eval')
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)

                tar_dir = os.path.join(imgs_dir, 'targets')
                gene_dir = os.path.join(imgs_dir, 'genes')
                real_dir = os.path.join(imgs_dir, 'reals')
                os.makedirs(tar_dir, exist_ok=True)
                os.makedirs(gene_dir, exist_ok=True)
                os.makedirs(real_dir, exist_ok=True)

                def save_image_png(img_array, filepath):
                    """Save image in PNG format (lossless).
                    
                    Args:
                        img_array: numpy array with values in [-1, 1] range, shape (H, W, C)
                        filepath: path to save the image
                    """
                    # Convert from [-1, 1] to [0, 255]
                    img_array = ((img_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    # Convert to PIL Image and save as PNG
                    img_pil = Image.fromarray(img_array)
                    img_pil.save(filepath, format='PNG')

                def row_stem(row):
                    return row.get('image_stem') or self._image_stem(row['image_path'])

                manifest_path = os.path.join(imgs_dir, 'manifest.csv')
                manifest_fields = [
                    'index', 'subject_id', 'eye_type',
                    'source_stem', 'target_stem',
                    'source_image_path', 'target_image_path',
                    'generated_path', 'target_saved_path', 'source_saved_path',
                    'source_pitch', 'source_yaw', 'target_pitch', 'target_yaw',
                ]
                manifest_pairs = getattr(self, 'eval_manifest_pairs', [])
                manifest_required = getattr(hps, 'dataset', 'magia') in ['columbia', 'xgaze']
                manifest_index = 0

                try:
                    with open(manifest_path, 'w', newline='') as manifest_file:
                        writer = csv.DictWriter(
                            manifest_file,
                            fieldnames=manifest_fields)
                        writer.writeheader()

                        i = 0
                        while True:
                            (real_imgs, target_imgs, fake_imgs,
                             a_r, a_t, labels_test, sides_test) = test_sess.run(
                                [self.x_test_r, self.x_test_t, x_fake,
                                 self.angles_test_r, self.angles_test_g, self.labels_test,
                                 self.sides_test])
                            if getattr(hps, 'dataset', 'magia') in ['magia', 'columbia', 'xgaze']:
                                a_t_for_name = np.degrees(a_t)
                                a_r_for_name = np.degrees(a_r)
                            else:
                                a_t_for_name = a_t * np.array([15, 10])
                                a_r_for_name = a_r * np.array([15, 10])
                            delta = angular_error(a_t_for_name, a_r_for_name)

                            for j in range(real_imgs.shape[0]):
                                dataset_name = getattr(hps, 'dataset', 'magia')
                                if manifest_index >= len(manifest_pairs):
                                    if manifest_required:
                                        raise RuntimeError(
                                            'manifest pair count is smaller than evaluated samples: '
                                            '%d pairs for sample index %d' % (
                                                len(manifest_pairs), manifest_index))
                                    source_row = target_row = None
                                else:
                                    source_row, target_row = manifest_pairs[manifest_index]

                                if dataset_name == 'xgaze':
                                    subject_id = source_row['subject_id']
                                    ref_side = source_row['eye_type']
                                    source_stem = row_stem(source_row)
                                    target_stem = row_stem(target_row)
                                    fn = f"[index={manifest_index:06d}][subject_id={subject_id}][ref_side={ref_side}][ref={source_stem}][target={target_stem}][origin_yaw={round(a_r_for_name[j][0])}][origin_pitch={round(a_r_for_name[j][1])}][target_yaw={round(a_t_for_name[j][0])}][target_pitch={round(a_t_for_name[j][1])}].png"
                                else:
                                    subject_id = labels_test[j] + 1
                                    fn = f"[subject_id={subject_id}][ref_side={sides_test[j].decode()}][origin_yaw={round(a_r_for_name[j][0])}][origin_pitch={round(a_r_for_name[j][1])}][target_yaw={round(a_t_for_name[j][0])}][target_pitch={round(a_t_for_name[j][1])}].png"
                                target_saved_path = os.path.join(tar_dir, fn)
                                generated_path = os.path.join(gene_dir, fn)
                                source_saved_path = os.path.join(real_dir, fn)
                                save_image_png(target_imgs[j], target_saved_path)
                                save_image_png(fake_imgs[j], generated_path)
                                save_image_png(real_imgs[j], source_saved_path)

                                if source_row is not None:
                                    writer.writerow({
                                        'index': manifest_index,
                                        'subject_id': source_row['subject_id'],
                                        'eye_type': source_row['eye_type'],
                                        'source_stem': row_stem(source_row),
                                        'target_stem': row_stem(target_row),
                                        'source_image_path': source_row['image_path'],
                                        'target_image_path': target_row['image_path'],
                                        'generated_path': generated_path,
                                        'target_saved_path': target_saved_path,
                                        'source_saved_path': source_saved_path,
                                        'source_pitch': source_row['pitch'],
                                        'source_yaw': source_row['yaw'],
                                        'target_pitch': target_row['pitch'],
                                        'target_yaw': target_row['yaw'],
                                    })
                                manifest_index += 1

                            i = i + 1
                except tf.errors.OutOfRangeError:
                    if manifest_required and manifest_index != len(manifest_pairs):
                        raise RuntimeError(
                            'manifest pair count does not match evaluated samples: '
                            '%d pairs, %d evaluated samples' % (
                                len(manifest_pairs), manifest_index))
                    logging.info("quanti_eval finished.")
