import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import sys

sys.path.append(str(Path(__file__).parent))
from models.cnn_model import MultiLabelCNN, EnsembleModel


class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
    
    def apply_gradients(self, grads_and_vars):
        result = self.optimizer.apply_gradients(grads_and_vars)
        self.step_count += 1
        if self.step_count % self.k == 0:
            for var, grad in grads_and_vars:
                if var.trainable:
                    var.assign_add(self.alpha * (self.optimizer.get_slot(var, 'm') - var))
        return result
    
    def get_config(self):
        return self.optimizer.get_config()


class ModelTrainer:
    def __init__(self, model_dir: str = 'outputs/models', random_seed: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
             epochs: int = 50, use_focal_loss: bool = True, class_weights: Optional[dict] = None,
             learning_rate: float = 0.001, callbacks: Optional[list] = None,
             base_models: List[str] = None, use_mixed_precision: bool = True,
             use_lookahead: bool = True) -> Dict:
        if base_models is None:
            base_models = ['ResNet50']
        
        models = []
        for base_model in base_models:
            model = MultiLabelCNN(num_classes=14, base_model=base_model, use_attention=True,
                                use_multi_scale=True, use_fpn=True, use_se=True)
            model.build(dropout_rate=0.5)
            model.compile_model(learning_rate=learning_rate, use_focal_loss=use_focal_loss,
                             class_weights=class_weights, label_smoothing=0.1,
                             use_mixed_precision=use_mixed_precision)
            
            if use_lookahead and hasattr(model.model.optimizer, 'apply_gradients'):
                original_optimizer = model.model.optimizer
                model.model.optimizer = Lookahead(original_optimizer, k=5, alpha=0.5)
            
            if callbacks is None:
                callbacks = self._get_default_callbacks()
            
            print(f"Training {base_model} with {model.model.count_params():,} parameters")
            history = model.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                                     callbacks=callbacks, verbose=1)
            
            best_model_path = self.model_dir / f'best_model_{base_model}.h5'
            model.model.save_weights(str(best_model_path))
            models.append(model.model)
        
        ensemble = EnsembleModel(models) if len(models) > 1 else None
        return {
            'models': models,
            'ensemble': ensemble,
            'best_model_path': self.model_dir / 'best_model.h5',
            'history': history.history if len(base_models) == 1 else None
        }
    
    def _get_default_callbacks(self) -> list:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'checkpoint_epoch_{epoch:02d}.h5'),
                save_best_only=True, monitor='val_auc', mode='max', save_weights_only=True, verbose=1
            ),
            keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10,
                                         restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                            min_lr=1e-7, verbose=1)
        ]
        try:
            from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
            callbacks.append(CosineDecayRestarts(initial_learning_rate=0.001, first_decay_steps=1000,
                                                t_mul=2.0, m_mul=0.5, alpha=0.0))
        except:
            pass
        return callbacks
