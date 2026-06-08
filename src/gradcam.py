import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GradCAMVisualizer:
    def __init__(self, model: tf.keras.Model, disease_labels: List[str], output_dir: str = 'outputs/gradcam'):
        self.model = model
        self.disease_labels = disease_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def make_gradcam_heatmap(self, img_array: np.ndarray, pred_index: int,
                            last_conv_layer_name: str = 'resnet50') -> np.ndarray:
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis])
        return (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
    
    def visualize_gradcam(self, image_path: str, pred_index: int, alpha: float = 0.4,
                         save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
        img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
        
        heatmap = self.make_gradcam_heatmap(img_array, pred_index)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = np.clip(heatmap_colored * alpha + img, 0, 255).astype(np.uint8)
        
        if save_path:
            cv2.imwrite(save_path, superimposed)
        return superimposed, heatmap_colored
    
    def visualize_multiple_diseases(self, image_path: str, predicted_classes: List[int],
                                   save_dir: Optional[str] = None):
        fig, axes = plt.subplots(1, len(predicted_classes) + 1, figsize=(5 * (len(predicted_classes) + 1), 5))
        img = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (224, 224)), cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        for idx, class_idx in enumerate(predicted_classes):
            _, heatmap = self.visualize_gradcam(image_path, class_idx, alpha=0.4)
            axes[idx + 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            axes[idx + 1].set_title(f'{self.disease_labels[class_idx]}', fontsize=12, fontweight='bold')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / f'gradcam_{Path(image_path).stem}.png', dpi=300, bbox_inches='tight')
        plt.close()
