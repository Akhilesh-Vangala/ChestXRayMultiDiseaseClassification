import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, List


class SEBlock(layers.Layer):
    def __init__(self, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.se = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(channels // self.reduction, activation='relu', use_bias=False),
            layers.Dense(channels, activation='sigmoid', use_bias=False)
        ])
        super().build(input_shape)
    
    def call(self, inputs):
        return inputs * tf.expand_dims(tf.expand_dims(self.se(inputs), 1), 1)


class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.query = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.key = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.value = layers.Conv2D(channels, 1, use_bias=False)
        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros', trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        batch_size, h, w, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[-1]
        q = tf.reshape(self.query(inputs), [batch_size, h * w, channels // 8])
        k = tf.reshape(self.key(inputs), [batch_size, h * w, channels // 8])
        v = tf.reshape(self.value(inputs), [batch_size, h * w, channels])
        attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        out = tf.reshape(tf.matmul(attention, v), [batch_size, h, w, channels])
        return self.gamma * out + inputs


class ChannelAttention(layers.Layer):
    def __init__(self, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = keras.Sequential([
            layers.Dense(channels // self.reduction, activation='relu', use_bias=False),
            layers.Dense(channels, activation='sigmoid', use_bias=False)
        ])
        super().build(input_shape)
    
    def call(self, inputs):
        avg_out = self.fc(self.avg_pool(inputs))
        max_out = self.fc(self.max_pool(inputs))
        return inputs * tf.expand_dims(tf.expand_dims(avg_out + max_out, 1), 1)


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        return inputs * self.conv(tf.concat([avg_out, max_out], axis=3))


class CBAM(layers.Layer):
    def __init__(self, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction=reduction)
        self.spatial_attention = SpatialAttention()
    
    def call(self, inputs):
        return self.spatial_attention(self.channel_attention(inputs))


class FPNBlock(layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, 1, padding='same')
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs):
        return tf.nn.relu(self.bn(self.conv(inputs)))


class MultiLabelCNN:
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 14, base_model: str = 'ResNet50',
                 use_attention: bool = True, use_multi_scale: bool = True,
                 use_fpn: bool = True, use_se: bool = True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale
        self.use_fpn = use_fpn
        self.use_se = use_se
        self.model = None
    
    def _get_base_model(self):
        models = {
            'ResNet50': keras.applications.ResNet50,
            'DenseNet121': keras.applications.DenseNet121,
            'EfficientNetB0': keras.applications.EfficientNetB0,
            'EfficientNetB3': keras.applications.EfficientNetB3
        }
        base = models[self.base_model_name](
            weights='imagenet', include_top=False, input_shape=self.input_shape
        )
        base.trainable = True
        return base
    
    def _build_fpn_features(self, base_output):
        fpn_features = []
        if hasattr(base_output, 'shape'):
            channels = base_output.shape[-1]
            fpn_features.append(FPNBlock(channels)(base_output))
            pooled = layers.AveragePooling2D(pool_size=(2, 2))(base_output)
            fpn_features.append(FPNBlock(channels // 2)(pooled))
            pooled = layers.AveragePooling2D(pool_size=(4, 4))(base_output)
            fpn_features.append(FPNBlock(channels // 4)(pooled))
        return fpn_features
    
    def _build_multi_scale_features(self, base_output):
        x1 = layers.GlobalAveragePooling2D()(base_output)
        x2 = layers.GlobalMaxPooling2D()(base_output)
        pooled = layers.AveragePooling2D(pool_size=(2, 2))(base_output)
        x3 = layers.GlobalAveragePooling2D()(pooled)
        x4 = layers.GlobalMaxPooling2D()(pooled)
        pooled2 = layers.AveragePooling2D(pool_size=(4, 4))(base_output)
        x5 = layers.GlobalAveragePooling2D()(pooled2)
        x6 = layers.GlobalMaxPooling2D()(pooled2)
        return layers.Concatenate()([x1, x2, x3, x4, x5, x6])
    
    def build(self, dropout_rate: float = 0.5) -> Model:
        base = self._get_base_model()
        inputs = keras.Input(shape=self.input_shape)
        x = base(inputs, training=True)
        
        if self.use_se:
            x = SEBlock(reduction=16)(x)
        
        if self.use_attention:
            x = CBAM(reduction=16)(x)
            x = SelfAttention()(x)
        
        if self.use_fpn:
            fpn_features = self._build_fpn_features(x)
            x = layers.Concatenate()([layers.GlobalAveragePooling2D()(f) for f in fpn_features])
        elif self.use_multi_scale:
            x = self._build_multi_scale_features(x)
        else:
            x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs, name='MultiLabelChestXRayCNN')
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001, use_focal_loss: bool = True,
                     class_weights: Optional[dict] = None, label_smoothing: float = 0.0,
                     use_mixed_precision: bool = True):
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
        
        try:
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        except AttributeError:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        loss = self._focal_loss(class_weights) if use_focal_loss else (
            keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing) if label_smoothing > 0
            else keras.losses.BinaryCrossentropy()
        )
        
        metrics = [
            keras.metrics.AUC(name='auc', multi_label=True),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='pr_auc', curve='PR', multi_label=True)
        ]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def _focal_loss(self, class_weights: Optional[dict] = None, alpha: float = 0.25, gamma: float = 2.0):
        def focal_loss_fn(y_true, y_pred):
            epsilon = keras.backend.epsilon()
            y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
            alpha_t = keras.backend.constant(list(class_weights.values()), dtype=tf.float32) if class_weights else alpha
            if class_weights:
                alpha_t = keras.backend.expand_dims(alpha_t, axis=0)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            return keras.backend.mean(-alpha_t * keras.backend.pow((1 - p_t), gamma) * keras.backend.log(p_t))
        return focal_loss_fn


class EnsembleModel:
    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
    
    def predict(self, x, use_tta: bool = False):
        if use_tta:
            predictions = []
            for model in self.models:
                preds = []
                augs = [
                    lambda img: img,
                    lambda img: tf.image.flip_left_right(img),
                    lambda img: tf.image.flip_up_down(img),
                    lambda img: tf.image.rot90(img),
                    lambda img: tf.image.rot90(img, k=2),
                    lambda img: tf.image.rot90(img, k=3)
                ]
                for aug in augs:
                    preds.append(model(aug(x), training=False))
                predictions.append(tf.reduce_mean(preds, axis=0))
        else:
            predictions = [model(x, training=False) for model in self.models]
        return tf.reduce_sum([w * p for w, p in zip(self.weights, predictions)], axis=0)
