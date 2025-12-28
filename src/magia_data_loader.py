# Dataloader.

import os
from pathlib import Path

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from loguru import logger


class ImageData(object):

    """ Dataloader for loading images. """

    def __init__(self,
            load_size,
            channels,
            data_path, # useless
            ids, # useless
            root_path: Path=Path("/home/18T/zhouhao/magia_gagen/EyeGazeRedirection/data/columbia/processed/"),
            metadata_path: Path=Path("metadata.csv"),
            pitch_scale: float=10.0,
            yaw_scale: float=15.0
        ):

        """ Init.

        Parameters
        ----------
        load_size: int, input image size.
        channels: int, number of channels.
        data_path: str, path of input images.
        ids: int, train/test split point.

        """

        self.load_size = load_size
        self.channels = channels
        self.root_path = Path(root_path)
        self.metadata_path = Path(metadata_path)
        self.pitch_scale = pitch_scale
        self.yaw_scale = yaw_scale

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g
    ):
        """ Process input images.

        Parameters
        ----------
        filename: str, path of input image.
        angles_r: list, gaze direction of input image.
        labels: int, subject id. (deprecated!)
        filename_t: str, path of target image.
        angles_g: list, gaze direction of target image.

        Returns
        -------
        image: tensor, float32, normalized input image.
        angles_r: angels_r.
        labels: labels.
        image_t: tensor, float32, normalized target image.
        angles_g: angles_g.

        """

        def _to_image(file_name):

            """ Load image, normalize it and convert it into tf.tensor.

            Parameters
            ----------
            file_name: str, image path.

            Returns
            -------
            img: tf.tensor, tf.float32. Image tensor.

            """

            x = tf.read_file(file_name)
            img = tf.image.decode_jpeg(x, channels=self.channels)
            img = tf.image.resize_images(img, [self.load_size, self.load_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0

            return img

        image = _to_image(filename)
        image_t = _to_image(filename_t)

        return image, angles_r, labels, image_t, angles_g

    def preprocess(self):

        metadata_file = self.root_path / self.metadata_path

        df = pd.read_csv(metadata_file)
        df["subject_id"] = df["subject_id"].astype(int)
        df.loc[df["eye_type"] == "right", "yaw"] = -df.loc[df["eye_type"] == "right", "yaw"]

        df_grouped = df.groupby(["subject_id", "eye_type", "split"])
        for (subject_id, eye_type, split), group in tqdm(df_grouped):
            len_group = len(group)
            logger.info(f"len_group: {len_group}")

            for i in range(len_group):
                row_reference = group.iloc[i]
                pitch_reference = row_reference["pitch"] / self.pitch_scale
                yaw_reference = row_reference["yaw"] / self.yaw_scale
                image_reference_path = self.root_path / row_reference["image_path"]
                
                for j in range(len_group):
                    row_generated = group.iloc[j]
                    pitch_generated = row_generated["pitch"] / self.pitch_scale
                    yaw_generated = row_generated["yaw"] / self.yaw_scale
                    image_generated_path = self.root_path / row_generated["image_path"]

                    if split == "train":
                        self.train_images.append(str(image_reference_path))
                        self.train_angles_r.append([pitch_reference, yaw_reference])
                        self.train_labels.append(subject_id - 1)
                        self.train_images_t.append(str(image_generated_path))
                        self.train_angles_g.append([pitch_generated, yaw_generated])
                    
                    if split == "test":
                        self.test_images.append(str(image_reference_path))
                        self.test_angles_r.append([pitch_reference, yaw_reference])
                        self.test_labels.append(subject_id - 1)
                        self.test_images_t.append(str(image_generated_path))
                        self.test_angles_g.append([pitch_generated, yaw_generated])
                    

if __name__ == "__main__":
    ImageData(load_size=64, channels=3).preprocess()
