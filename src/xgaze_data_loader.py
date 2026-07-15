# Dataloader for XGaze metadata.

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


class ImageData(object):

    """Online ref-target dataloader for XGaze metadata."""

    def __init__(
            self,
            load_size,
            channels,
            data_path=None,
            ids=None,
            metadata_path=None,
            image_dir=None,
            max_train_samples=None,
            max_eval_samples=20,
            min_lightness=None,
            train_subjects=60,
            val_subjects=10,
            eval_seed=0):

        self.load_size = load_size
        self.channels = channels
        self.data_path = Path(data_path) if data_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.min_lightness = min_lightness
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.eval_seed = eval_seed

        self.train_records = []
        self.valid_records = []
        self.test_records = []
        self.valid_pairs = []
        self.test_pairs = []
        self.train_size = 0

    def _paths(self):
        if self.metadata_path is not None:
            metadata_file = self.metadata_path
            image_dir = self.image_dir or metadata_file.parent
        elif self.data_path is not None and self.data_path.suffix.lower() == ".csv":
            metadata_file = self.data_path
            image_dir = self.image_dir or metadata_file.parent
        elif self.data_path is not None:
            metadata_file = self.data_path / "metadata.csv"
            image_dir = self.image_dir or self.data_path
        else:
            raise ValueError("xgaze dataset requires --data_path or --metadata_path")

        return metadata_file, image_dir

    @staticmethod
    def _resolve_path(image_dir, value):
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str(image_dir / path)

    @staticmethod
    def _sample_preview(records, max_samples, seed):
        if max_samples is None or len(records) <= max_samples:
            return records
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        return [records[index] for index in indices[:max_samples]]

    @staticmethod
    def _build_groups(records):
        groups = {}
        for index, record in enumerate(records):
            key = (record["subject_id"], record["eye_type"])
            groups.setdefault(key, []).append(index)
        return groups

    @classmethod
    def _fixed_pairs(cls, records, seed):
        groups = cls._build_groups(records)
        rng = random.Random(seed)
        pairs = []
        for index, ref in enumerate(records):
            group = groups[(ref["subject_id"], ref["eye_type"])]
            candidates = [candidate for candidate in group if candidate != index]
            target = records[rng.choice(candidates)] if candidates else ref
            pairs.append((ref, target))
        return pairs

    @staticmethod
    def _example(ref, target):
        return (
            ref["image_path"],
            np.asarray([ref["yaw"], ref["pitch"]], dtype=np.float32),
            ref["subject_id"],
            target["image_path"],
            np.asarray([target["yaw"], target["pitch"]], dtype=np.float32),
            ref["eye_type"],
        )

    def preprocess(self):
        metadata_file, image_dir = self._paths()
        required_columns = {"image_path", "subject_id", "eye_type", "pitch", "yaw"}
        if self.min_lightness is not None:
            required_columns.add("lightness")

        df = pd.read_csv(metadata_file, usecols=sorted(required_columns)).sort_values(
            by=["subject_id", "eye_type"])
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError("metadata missing columns: %s" % sorted(missing_columns))

        subjects = sorted(df["subject_id"].unique())
        if len(subjects) < self.train_subjects + self.val_subjects:
            raise ValueError("xgaze requires at least %d subjects, got %d" % (
                self.train_subjects + self.val_subjects, len(subjects)))

        subject_to_split = {}
        for index, subject in enumerate(subjects):
            if index < self.train_subjects:
                split = "train"
            elif index < self.train_subjects + self.val_subjects:
                split = "val"
            else:
                split = "test"
            subject_to_split[subject] = split
        df["split"] = df["subject_id"].map(subject_to_split)

        if self.min_lightness is not None:
            df = df[df["lightness"] > float(self.min_lightness)].copy()

        df["image_path"] = df["image_path"].apply(
            lambda value: self._resolve_path(image_dir, value))

        columns_to_keep = ["subject_id", "eye_type", "split", "pitch", "yaw", "image_path"]
        split_records = {}
        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]
            split_records[split_name] = split_df[columns_to_keep].to_dict("records")

        train_records = split_records["train"]
        if self.max_train_samples is not None:
            train_records = train_records[:self.max_train_samples]

        valid_records = (
            self._sample_preview(split_records["val"], self.max_eval_samples, self.eval_seed + 43) +
            self._sample_preview(train_records, self.max_eval_samples, self.eval_seed + 42))

        self.train_records = train_records
        self.valid_records = valid_records
        self.test_records = split_records["test"]
        self.valid_pairs = self._fixed_pairs(self.valid_records, self.eval_seed)
        self.test_pairs = self._fixed_pairs(self.test_records, self.eval_seed)
        self.train_size = len(self.train_records)

        print("\nFinished preprocessing the xgaze dataset...")
        print("subjects: train=%d, val=%d, test=%d" % (
            self.train_subjects,
            self.val_subjects,
            len(subjects) - self.train_subjects - self.val_subjects))
        print("records: train=%d, valid=%d, test=%d" % (
            len(self.train_records), len(self.valid_records), len(self.test_records)))

    def _train_generator(self):
        groups = self._build_groups(self.train_records)
        rng = random.Random()
        while True:
            indices = list(range(len(self.train_records)))
            rng.shuffle(indices)
            for index in indices:
                ref = self.train_records[index]
                group = groups[(ref["subject_id"], ref["eye_type"])]
                candidates = [candidate for candidate in group if candidate != index]
                target = self.train_records[rng.choice(candidates)] if candidates else ref
                yield self._example(ref, target)

    def _fixed_generator(self, pairs, repeat):
        while True:
            for ref, target in pairs:
                yield self._example(ref, target)
            if not repeat:
                break

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g,
        side
    ):
        def _to_image(file_name):
            x = tf.read_file(file_name)
            img = tf.image.decode_jpeg(x, channels=self.channels)
            img = tf.image.resize_images(img, [self.load_size, self.load_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            return img

        image = _to_image(filename)
        image_t = _to_image(filename_t)

        return image, angles_r, labels, image_t, angles_g, side

    def make_dataset(self, split, batch_size):
        output_types = (tf.string, tf.float32, tf.string, tf.string, tf.float32, tf.string)
        output_shapes = ((), (2,), (), (), (2,), ())

        if split == "train":
            dataset = tf.data.Dataset.from_generator(
                self._train_generator,
                output_types=output_types,
                output_shapes=output_shapes)
        elif split == "val":
            dataset = tf.data.Dataset.from_generator(
                lambda: self._fixed_generator(self.valid_pairs, repeat=True),
                output_types=output_types,
                output_shapes=output_shapes)
        elif split == "test":
            dataset = tf.data.Dataset.from_generator(
                lambda: self._fixed_generator(self.test_pairs, repeat=False),
                output_types=output_types,
                output_shapes=output_shapes)
        else:
            raise ValueError("unknown split: %s" % split)

        return dataset.map(self.image_processing).batch(batch_size)
