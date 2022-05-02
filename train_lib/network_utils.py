import tensorflow as tf

def pressure_matching_network(sf_shape, nfft, filter_shape):

    # Encoder
    input_layer = tf.keras.layers.Input(shape=(sf_shape, nfft, 1))
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(input_layer)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Flatten()(x)

    # Bottleneck
    depth_decoder = 4
    bottleneck_dim_filters = 8
    bottleneck_dim_freq = 8
    x = tf.keras.layers.Dense(bottleneck_dim_filters*bottleneck_dim_freq)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Reshape((bottleneck_dim_filters, bottleneck_dim_freq,1))(x)

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, 1, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 1, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 1, padding='same')(x)
    x = tf.keras.layers.PReLU()(x)

    # Output
    out = tf.keras.layers.Conv2DTranspose(1, 3, 1, padding='same')(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=out)