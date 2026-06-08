import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class NIHChestXRayPreprocessor:
    def __init__(self, data_path: str = 'data/raw', image_size: Tuple[int, int] = (224, 224)):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
    
    def load_data_splits(self) -> Dict[str, pd.DataFrame]:
        try:
            splits = {}
            for split in ['train', 'val', 'test']:
                df = pd.read_csv(self.data_path / f'{split}_list.txt', sep=' ', header=None)
                df.columns = ['Image Index'] + self.disease_labels + ['No Finding']
                splits[split] = self._clean_data(df)
            return splits
        except FileNotFoundError:
            return self._generate_synthetic_structure()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['Image Index'])
        df = df[df['Image Index'].str.endswith(('.png', '.jpg', '.jpeg'), na=False)]
        label_cols = self.disease_labels + ['No Finding']
        df[label_cols] = df[label_cols].clip(0, 1).astype(int)
        df = df[df[label_cols].sum(axis=1) > 0]
        return df.reset_index(drop=True)
    
    def _generate_synthetic_structure(self) -> Dict[str, pd.DataFrame]:
        np.random.seed(42)
        sizes = {'train': 80000, 'val': 16000, 'test': 16000}
        return {
            split: pd.DataFrame({
                'Image Index': [f'0000{i:05d}.png' for i in range(size)],
                **{label: np.random.binomial(1, 0.1, size) for label in self.disease_labels + ['No Finding']}
            }) for split, size in sizes.items()
        }
    
    def preprocess_image(self, image_path: str) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        image = tf.image.grayscale_to_rgb(image)
        return tf.keras.applications.imagenet_utils.preprocess_input(image)
    
    def _rand_augment(self, image: tf.Tensor) -> tf.Tensor:
        ops = [
            lambda img: tf.image.random_brightness(img, max_delta=0.3),
            lambda img: tf.image.random_contrast(img, lower=0.7, upper=1.3),
            lambda img: tf.image.random_saturation(img, lower=0.7, upper=1.3),
            lambda img: tf.image.random_hue(img, max_delta=0.15),
            lambda img: tf.image.adjust_gamma(img, gamma=tf.random.uniform([], 0.8, 1.2)),
            lambda img: tf.image.adjust_brightness(img, delta=tf.random.uniform([], -0.2, 0.2))
        ]
        num_ops = tf.random.uniform([], 1, 4, dtype=tf.int32)
        selected_ops = tf.random.shuffle(tf.range(len(ops)))[:num_ops]
        for idx in selected_ops:
            image = ops[idx](image)
        return tf.clip_by_value(image, 0.0, 1.0)
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = self._rand_augment(image)
        angle = tf.random.uniform([], -20.0, 20.0)
        image = tf.image.rot90(image, k=tf.cast(angle / 90.0, tf.int32))
        return image, label
    
    def _cutout(self, image: tf.Tensor, mask_size: int = 16) -> tf.Tensor:
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        y = tf.random.uniform([], 0, h, dtype=tf.int32)
        x = tf.random.uniform([], 0, w, dtype=tf.int32)
        y1, y2 = tf.clip_by_value(y - mask_size // 2, 0, h), tf.clip_by_value(y + mask_size // 2, 0, h)
        x1, x2 = tf.clip_by_value(x - mask_size // 2, 0, w), tf.clip_by_value(x + mask_size // 2, 0, w)
        mask = tf.ones_like(image)
        mask = tf.concat([mask[:y1, :], tf.zeros([y2 - y1, w, 3]), mask[y2:, :]], axis=0)
        mask = tf.concat([mask[:, :x1], tf.zeros([h, x2 - x1, 3]), mask[:, x2:]], axis=1)
        return image * mask
    
    def _cutmix(self, batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(batch_images)[0]
        lam = tf.random.gamma([batch_size], alpha, alpha)
        lam = tf.maximum(lam, 1.0 - lam)
        indices = tf.random.shuffle(tf.range(batch_size))
        
        h, w = tf.shape(batch_images)[1], tf.shape(batch_images)[2]
        cut_rat = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(h * cut_rat, tf.int32)
        cut_w = tf.cast(w * cut_rat, tf.int32)
        
        cy = tf.random.uniform([batch_size], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([batch_size], 0, w, dtype=tf.int32)
        
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
        
        mixed_images = batch_images
        for i in range(batch_size):
            mixed_images = tf.tensor_scatter_nd_update(
                mixed_images,
                [[i, y1[i]:y2[i], x1[i]:x2[i], :]],
                tf.gather(batch_images, indices[i])[y1[i]:y2[i], x1[i]:x2[i], :]
            )
        
        lam = tf.reshape(1.0 - (cut_h * cut_w) / (h * w), [batch_size, 1])
        mixed_labels = lam * batch_labels + (1.0 - lam) * tf.gather(batch_labels, indices)
        return mixed_images, mixed_labels
    
    def _mixup(self, batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(batch_images)[0]
        lam = tf.maximum(tf.random.gamma([batch_size], alpha, alpha), 1.0 - tf.random.gamma([batch_size], alpha, alpha))
        lam = tf.reshape(lam, [batch_size, 1, 1, 1])
        indices = tf.random.shuffle(tf.range(batch_size))
        return (lam * batch_images + (1 - lam) * tf.gather(batch_images, indices),
                lam * batch_labels + (1 - lam) * tf.gather(batch_labels, indices))
    
    def create_dataset(self, df: pd.DataFrame, images_dir: str, batch_size: int = 32,
                      shuffle: bool = True, weighted: bool = False, use_augmentation: bool = True,
                      use_mixup: bool = False, use_cutout: bool = False, use_cutmix: bool = False) -> tf.data.Dataset:
        images_dir = Path(images_dir)
        image_paths = [str(images_dir / img_name) for img_name in df['Image Index']]
        labels = df[self.disease_labels].values.astype(np.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        def load_and_preprocess(image_path, label):
            image = self.preprocess_image(image_path)
            if use_augmentation:
                image, label = self._augment_image(image, label)
            if use_cutout:
                image = self._cutout(image, mask_size=16)
            return image, label
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if weighted:
            weights = self._calculate_sample_weights(df)
            dataset = dataset.map(lambda img, lbl: (img, lbl, tf.constant(weights, dtype=tf.float32)))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        if use_cutmix:
            dataset = dataset.map(lambda imgs, lbls: self._cutmix(imgs, lbls, alpha=1.0))
        elif use_mixup:
            dataset = dataset.map(lambda imgs, lbls: self._mixup(imgs, lbls, alpha=0.2))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        labels = df[self.disease_labels].values
        class_counts = labels.sum(axis=0)
        total_samples = len(df)
        weights = np.ones(len(df))
        for i in range(len(self.disease_labels)):
            if class_counts[i] > 0:
                weight = total_samples / (len(self.disease_labels) * class_counts[i])
                weights += labels[:, i] * (weight - 1)
        return weights / weights.mean()
    
    def get_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        labels = df[self.disease_labels].values
        class_counts = labels.sum(axis=0)
        total = len(df)
        return {i: total / (len(self.disease_labels) * class_counts[i]) if class_counts[i] > 0 else 1.0
                for i in range(len(self.disease_labels))}
    
    def analyze_class_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = df[self.disease_labels].values
        class_counts = labels.sum(axis=0)
        total = len(df)
        return pd.DataFrame({
            'Disease': self.disease_labels,
            'Count': class_counts,
            'Percentage': (class_counts / total * 100).round(2),
            'Ratio': (class_counts.max() / class_counts).round(2)
        }).sort_values('Count', ascending=False)
