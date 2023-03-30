import tensorflow as tf
from src.islr.models.utils import get_activation
from src.islr.models.config import INIT_HE_UNIFORM, INIT_GLOROT_UNIFORM, INIT_ZEROS


class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name, model_config):
        super(LandmarkEmbedding, self).__init__(name=f"{name}_embedding")
        self.units = units
        self.activation_fn = get_activation(model_config["ACTIVATION_FN"])

    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f"{self.name}_empty_embedding",
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.units,
                    name=f"{self.name}_dense_1",
                    use_bias=False,
                    kernel_initializer=INIT_GLOROT_UNIFORM,
                    activation=self.activation_fn,
                ),
                tf.keras.layers.Dense(
                    self.units,
                    name=f"{self.name}_dense_2",
                    use_bias=False,
                    kernel_initializer=INIT_HE_UNIFORM,
                ),
            ],
            name=f"{self.name}_dense",
        )

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )


class EmbeddingCustom(tf.keras.Model):
    def __init__(self, model_config, input_size):
        super(EmbeddingCustom, self).__init__()
        self.model_config = model_config
        self.input_size = input_size
        self.activation_fn = get_activation(model_config["ACTIVATION_FN"])

    def get_diffs(self, l):
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0, 1, 3, 2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, self.input_size, S * S])
        return diffs

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(
            self.input_size + 1,
            self.model_config["UNITS"],
            embeddings_initializer=INIT_ZEROS,
        )
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(
            self.model_config["LIPS_UNITS"], "lips", self.model_config
        )
        self.left_hand_embedding = LandmarkEmbedding(
            self.model_config["HANDS_UNITS"], "left_hand", self.model_config
        )
        self.right_hand_embedding = LandmarkEmbedding(
            self.model_config["HANDS_UNITS"], "right_hand", self.model_config
        )
        self.pose_embedding = LandmarkEmbedding(
            self.model_config["POSE_UNITS"], "pose", self.model_config
        )
        # Landmark Weights
        self.landmark_weights = tf.Variable(
            tf.zeros([4], dtype=tf.float32), name="landmark_weights"
        )
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.model_config["UNITS"],
                    name="fully_connected_1",
                    use_bias=False,
                    kernel_initializer=INIT_GLOROT_UNIFORM,
                    activation=self.activation_fn,
                ),
                tf.keras.layers.Dense(
                    self.model_config["UNITS"],
                    name="fully_connected_2",
                    use_bias=False,
                    kernel_initializer=INIT_HE_UNIFORM,
                ),
            ],
            name="fc",
        )

    def call(
        self,
        lips0,
        left_hand0,
        right_hand0,
        pose0,
        non_empty_frame_idxs,
        training=False,
    ):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Right Hand
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack(
            (lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding),
            axis=3,
        )
        # Merge Landmarks with trainable attention weights
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            self.input_size,
            tf.cast(
                non_empty_frame_idxs
                / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True)
                * self.input_size,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)

        return x
