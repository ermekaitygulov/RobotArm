import tensorflow as tf

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def take_vector_elements(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.
    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.
    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))

def config_gpu():
    tf.debugging.set_log_device_placement(False)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
