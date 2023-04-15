import tensorflow as tf


# based on: https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer
# replaced softmax with softmax layer to support masked softmax
def scaled_dot_product(q, k, v, softmax, attention_mask):
    # calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)

    z = tf.matmul(softmax, v)
    # shape: (m,Tx,depth), same shape as q,k,v
    return z


def get_activation(activation_name):
    """
    Get a TensorFlow activation function from a string name.

    Parameters:
        activation_name (str): Name of the activation function (e.g. "relu").

    Returns:
        The TensorFlow activation function corresponding to the given name.
    """
    return getattr(tf.keras.activations, activation_name)


def SparseCategoricalCrossentropyLS(y_true, y_pred, num_classes=250, smoothing=0.25):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, num_classes, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=smoothing
    )
