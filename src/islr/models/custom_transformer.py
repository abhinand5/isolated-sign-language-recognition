import tensorflow as tf
import tensorflow_addons as tfa

from src.islr.models.embedding import EmbeddingCustom
from src.islr.data_processing.feature_extraction import extra_features
from src.islr.models.utils import (
    scaled_dot_product,
    get_activation,
    SparseCategoricalCrossentropyLS,
)
from src.islr.models.config import INIT_HE_UNIFORM, INIT_GLOROT_UNIFORM
from src.islr.logging import get_logger

logger = get_logger(__name__)


class MultiHeadAttentionCustom(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttentionCustom, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


# Full Transformer
class TransformerCustom(tf.keras.Model):
    def __init__(self, num_blocks, model_config):
        super(TransformerCustom, self).__init__(name="transformer")
        self.num_blocks = num_blocks
        self.model_config = model_config
        self.activation_fn = get_activation(model_config["ACTIVATION_FN"])

    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.model_config["LAYER_NORM_EPS"]
                )
            )
            # Multi Head Attention
            self.mhas.append(
                MultiHeadAttentionCustom(
                    self.model_config["UNITS"], self.model_config["MHA_HEADS"]
                )
            )
            # Second Layer Normalisation
            self.ln_2s.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.model_config["LAYER_NORM_EPS"]
                )
            )
            # Multi Layer Perception
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            self.model_config["UNITS"] * self.model_config["MLP_RATIO"],
                            activation=self.activation_fn,
                            kernel_initializer=INIT_GLOROT_UNIFORM,
                        ),
                        tf.keras.layers.Dropout(self.model_config["MLP_DROPOUT_RATIO"]),
                        tf.keras.layers.Dense(
                            self.model_config["UNITS"],
                            kernel_initializer=INIT_HE_UNIFORM,
                        ),
                    ]
                )
            )

    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2

        return x


# Full Transformer without LayerNorm
class TransformerCustomNoLN(tf.keras.Model):
    def __init__(self, num_blocks, model_config):
        super(TransformerCustomNoLN, self).__init__(name="transformer")
        self.num_blocks = num_blocks
        self.model_config = model_config
        self.activation_fn = get_activation(model_config["ACTIVATION_FN"])

    def build(self, input_shape):
        self.mhas = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(
                MultiHeadAttentionCustom(
                    self.model_config["UNITS"], self.model_config["MHA_HEADS"]
                )
            )
            # Multi Layer Perception
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            self.model_config["UNITS"] * self.model_config["MLP_RATIO"],
                            activation=self.activation_fn,
                            kernel_initializer=INIT_GLOROT_UNIFORM,
                        ),
                        tf.keras.layers.Dropout(self.model_config["MLP_DROPOUT_RATIO"]),
                        tf.keras.layers.Dense(
                            self.model_config["UNITS"],
                            kernel_initializer=INIT_HE_UNIFORM,
                        ),
                    ]
                )
            )

    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)

        return x


def get_model(data_config, model_config, feature_stats):
    # Inputs
    frames = tf.keras.layers.Input(
        [data_config["INPUT_SIZE"], extra_features.N_COLS, data_config["N_DIMS"]],
        dtype=tf.float32,
        name="frames",
    )
    non_empty_frame_idxs = tf.keras.layers.Input(
        [data_config["INPUT_SIZE"]], dtype=tf.float32, name="non_empty_frame_idxs"
    )
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)

    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(
        x, [0, 0, 0, 0], [-1, data_config["INPUT_SIZE"], extra_features.N_COLS, 2]
    )
    # LIPS
    lips = tf.slice(
        x, [0, 0, extra_features.LIPS_START, 0], [-1, data_config["INPUT_SIZE"], 40, 2]
    )
    lips = tf.where(
        tf.math.equal(lips, 0.0),
        0.0,
        (lips - feature_stats["LIPS_MEAN"]) / feature_stats["LIPS_STD"],
    )
    lips = tf.reshape(lips, [-1, data_config["INPUT_SIZE"], 40 * 2])
    # LEFT HAND
    left_hand = tf.slice(x, [0, 0, 40, 0], [-1, data_config["INPUT_SIZE"], 21, 2])
    left_hand = tf.where(
        tf.math.equal(left_hand, 0.0),
        0.0,
        (left_hand - feature_stats["LEFT_HANDS_MEAN"])
        / feature_stats["LEFT_HANDS_STD"],
    )
    left_hand = tf.reshape(left_hand, [-1, data_config["INPUT_SIZE"], 21 * 2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0, 0, 61, 0], [-1, data_config["INPUT_SIZE"], 21, 2])
    right_hand = tf.where(
        tf.math.equal(right_hand, 0.0),
        0.0,
        (right_hand - feature_stats["RIGHT_HANDS_MEAN"])
        / feature_stats["RIGHT_HANDS_STD"],
    )
    right_hand = tf.reshape(right_hand, [-1, data_config["INPUT_SIZE"], 21 * 2])
    # POSE
    pose = tf.slice(x, [0, 0, 82, 0], [-1, data_config["INPUT_SIZE"], 10, 2])
    pose = tf.where(
        tf.math.equal(pose, 0.0),
        0.0,
        (pose - feature_stats["POSE_MEAN"]) / feature_stats["POSE_STD"],
    )
    pose = tf.reshape(pose, [-1, data_config["INPUT_SIZE"], 10 * 2])

    x = lips, left_hand, right_hand, pose

    x = EmbeddingCustom(model_config, data_config["INPUT_SIZE"])(
        lips, left_hand, right_hand, pose, non_empty_frame_idxs
    )

    # Encoder Transformer Blocks
    if model_config["ADD_LAYER_NORM"]:
        x = TransformerCustom(model_config["NUM_BLOCKS"], model_config)(x, mask)
    else:
        x = TransformerCustomNoLN(model_config["NUM_BLOCKS"], model_config)(x, mask)

    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classification Layer
    x = tf.keras.layers.Dropout(model_config["CLASSIFIER_DROPOUT_RATIO"])(x)
    x = tf.keras.layers.Dense(
        data_config["NUM_CLASSES"],
        activation=tf.keras.activations.softmax,
        kernel_initializer=INIT_GLOROT_UNIFORM,
    )(x)

    outputs = x

    # Create Tensorflow Model
    model = tf.keras.models.Model(
        inputs=[frames, non_empty_frame_idxs], outputs=outputs
    )

    # Simple Categorical Crossentropy Loss
    if model_config["LABEL_SMOOTHING"] is not None:
        logger.info(
            "Choosing Label Smoothing Loss function -> SparseCategoricalCrossentropyLS"
        )
        loss = SparseCategoricalCrossentropyLS
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # lr_schedule = (
    #     tf.keras.optimizers.schedules.CosineDecayRestarts(
    #         name="CosineDecayRestarts",
    #         initial_learning_rate=INIT_LR,
    #         first_decay_steps=first_decay_steps,
    #         t_mul=1.0,
    #         m_mul=0.9,
    #         alpha=0.05)
    #     )

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     INIT_LR,
    #     decay_steps=(86079 // BATCH_SIZE) * 3,
    #     decay_rate=0.95,
    #     staircase=False)

    # Adam Optimizer with weight decay
    # optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)
    # weight_decay=7.12332615534032e-06, clipnorm=1.0)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=model_config["INIT_LR"],
        weight_decay=model_config["WT_DECAY"],
        clipnorm=1.0,
    )

    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top_10_acc"),
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model
