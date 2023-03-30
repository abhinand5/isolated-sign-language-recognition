import tensorflow as tf
from src.islr.data_processing.feature_extraction import extra_features


class PreprocessLayer(tf.keras.layers.Layer):
    """
    A Tensorflow layer for preprocessing video data for use in deep learning models.

    This layer applies several preprocessing steps to the input data, including filtering out frames with empty hand data,
    gathering only relevant landmark columns, padding and downsampling to fit within a specified input size,
    and mean pooling to reduce the size of the data for improved model performance.

    Parameters:
    -----------
    None

    Methods:
    --------
    pad_edge(t: tf.Tensor, repeats: int, side: str) -> tf.Tensor:
        Helper function for padding the input tensor on the left or right edge.

    call(data0: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        Applies preprocessing steps to the input data and returns the processed data and non-empty frames indices.

    Returns:
    --------
    A `PreprocessLayer` object.
    """

    def __init__(self, input_size, n_dims):
        super(PreprocessLayer, self).__init__()
        self.input_size = input_size
        self.n_dims = n_dims

    def pad_edge(self, t, repeats, side):
        if side == "LEFT":
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == "RIGHT":
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]

        # Filter Out Frames With Empty Hand Data
        frames_hands_nansum = tf.experimental.numpy.nanmean(
            tf.gather(data0, extra_features.HAND_IDXS0, axis=1), axis=[1, 2]
        )
        non_empty_frames_idxs = tf.where(frames_hands_nansum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)

        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32)

        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]

        # Gather Relevant Landmark Columns
        data = tf.gather(data, extra_features.LANDMARK_IDXS0, axis=1)

        # Video fits in INPUT_SIZE
        if N_FRAMES < self.input_size:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(
                non_empty_frames_idxs,
                [[0, self.input_size - N_FRAMES]],
                constant_values=-1,
            )
            # Pad Data With Zeros
            data = tf.pad(
                data,
                [[0, self.input_size - N_FRAMES], [0, 0], [0, 0]],
                constant_values=0,
            )
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < self.input_size**2:
                repeats = tf.math.floordiv(self.input_size * self.input_size, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(
                    non_empty_frames_idxs, repeats=repeats, axis=0
                )

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), self.input_size)
            if tf.math.mod(len(data), self.input_size) > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * self.input_size) - len(data)
            else:
                pad_size = (pool_size * self.input_size) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(
                self.input_size, 2
            )
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(
                self.input_size, 2
            )
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, "LEFT")
            data = self.pad_edge(data, pad_right, "RIGHT")

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(
                non_empty_frames_idxs, pad_left, "LEFT"
            )
            non_empty_frames_idxs = self.pad_edge(
                non_empty_frames_idxs, pad_right, "RIGHT"
            )

            # Reshape to Mean Pool
            data = tf.reshape(
                data,
                [
                    self.input_size,
                    -1,
                    extra_features.N_COLS,
                    self.n_dims,
                ],
            )
            non_empty_frames_idxs = tf.reshape(
                non_empty_frames_idxs, [self.input_size, -1]
            )

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(
                non_empty_frames_idxs, axis=1
            )

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)

            return data, non_empty_frames_idxs
