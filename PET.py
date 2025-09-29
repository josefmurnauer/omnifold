import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------------
# Custom Layers
# -------------------------------

class ParticleMask(layers.Layer):
    """Mask particles where all features are zero."""
    def call(self, x):
        return tf.reduce_any(tf.not_equal(x, 0), axis=-1)  # (batch, seq)

class MaskedAveragePooling(layers.Layer):
    """Average pooling over masked sequence."""
    def call(self, x, mask):
        mask = tf.cast(mask, tf.float32)[..., None]
        x = x * mask
        return tf.reduce_sum(x, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)

class LayerScale(layers.Layer):
    def __init__(self, projection_dim, init_value=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
            name='gamma'
        )

    def call(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * tf.cast(mask[..., None], tf.float32)
        return x * self.gamma

class KNNLayer(layers.Layer):
    """Compute local KNN features."""
    def __init__(self, K, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.projection_dim = projection_dim
        self.dense1 = layers.Dense(2 * projection_dim, activation='gelu')
        self.dense2 = layers.Dense(projection_dim, activation='gelu')

    def call(self, points, features):
        # points: (batch, P, 2), features: (batch, P, C)
        # Compute pairwise distance
        r = tf.reduce_sum(points * points, axis=-1, keepdims=True)
        m = tf.matmul(points, points, transpose_b=True)
        D = tf.abs(r - 2 * m + tf.transpose(r, perm=(0, 2, 1)))

        # KNN indices
        indices = tf.nn.top_k(-D, k=self.K + 1).indices[:, :, 1:]  # (batch, P, K)

        # Gather KNN features
        batch_size = tf.shape(features)[0]
        num_points = tf.shape(features)[1]
        batch_indices = tf.reshape(tf.range(batch_size), (-1, 1, 1))
        batch_indices = tf.tile(batch_indices, (1, num_points, self.K))
        knn_indices = tf.stack([batch_indices, indices], axis=-1)
        knn_features = tf.gather_nd(features, knn_indices)  # (batch, P, K, C)

        # Center features broadcast
        center_features = tf.expand_dims(features, 2)
        center_features = tf.broadcast_to(center_features, tf.shape(knn_features))

        # Concatenate differences
        local = tf.concat([knn_features - center_features, center_features], axis=-1)
        local = self.dense1(local)
        local = self.dense2(local)
        local = tf.reduce_mean(local, axis=2)  # average over K neighbors
        return local

# -------------------------------
# PET Model
# -------------------------------

class PET(Model):
    def __init__(self, num_feat, num_part=12, projection_dim=32, num_heads=2,
                 num_transformer=2, local=True, K=3, layer_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.num_feat = num_feat
        self.num_part = num_part
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.local = local
        self.K = K
        self.layer_scale = layer_scale

        # Input encoding
        self.input_dense1 = layers.Dense(2 * projection_dim, activation='gelu')
        self.input_dense2 = layers.Dense(projection_dim, activation='gelu')

        # Local KNN layer
        if self.local:
            self.knn_layer = KNNLayer(K=K, projection_dim=projection_dim)

        # Transformer blocks
        self.norm_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_transformer * 2)]
        self.mha_layers = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads)
                           for _ in range(num_transformer)]
        self.ffn_dense1 = [layers.Dense(2 * projection_dim, activation='gelu') for _ in range(num_transformer)]
        self.ffn_dense2 = [layers.Dense(projection_dim) for _ in range(num_transformer)]
        if self.layer_scale:
            self.layer_scales = [LayerScale(projection_dim) for _ in range(num_transformer * 2)]
        else:
            self.layer_scales = [None] * (num_transformer * 2)

        # Class token
        self.class_token = self.add_weight(
            shape=(1, projection_dim),
            initializer='zeros',
            trainable=True,
            name='class_token'
        )

        # Output head
        self.out_dense1 = layers.Dense(64, activation='relu')
        self.out_dense2 = layers.Dense(1, activation=None)

        # Masking
        self.particle_mask = ParticleMask()
        self.pool = MaskedAveragePooling()

    def call(self, x, training=False):
        mask = self.particle_mask(x)  # (batch, P)

        # Input encoding
        encoded = self.input_dense1(x)
        encoded = self.input_dense2(encoded)

        # Local KNN
        if self.local:
            points = x[:, :, 1:3]  # assuming first 2 features are coordinates
            local_features = self.knn_layer(points, encoded)
            encoded += local_features

        # Apply mask
        encoded *= tf.cast(mask[..., None], tf.float32)

        # Transformer blocks
        for i in range(self.num_transformer):
            x1 = self.norm_layers[2 * i](encoded)
            attn = self.mha_layers[i](x1, x1)
            if self.layer_scale:
                attn = self.layer_scales[2 * i](attn, mask)
            x2 = attn + encoded

            x3 = self.norm_layers[2 * i + 1](x2)
            x3 = self.ffn_dense1[i](x3)
            x3 = self.ffn_dense2[i](x3)
            if self.layer_scale:
                x3 = self.layer_scales[2 * i + 1](x3, mask)
            encoded = x3 + x2
            encoded *= tf.cast(mask[..., None], tf.float32)

        # Add class token
        batch_size = tf.shape(encoded)[0]
        class_tokens = tf.tile(self.class_token[None, :, :], [batch_size, 1, 1])
        encoded = tf.concat([class_tokens, encoded], axis=1)

        # Pooling over sequence (exclude class token?)
        pooled = self.pool(encoded[:, 1:], mask)  # average pooling over particles

        # Output
        x = self.out_dense1(pooled)
        out = self.out_dense2(x)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_feat": self.num_feat,
            "num_part": self.num_part,
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "num_transformer": self.num_transformer,
            "local": self.local,
            "K": self.K,
            "layer_scale": self.layer_scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)