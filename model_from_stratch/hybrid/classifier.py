import tensorflow as tf

CLASSES = 3
input_size = None

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(608, 608, 3)),
    tf.keras.layers.AveragePooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu'),
    tf.keras.layers.AveragePooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, kernel_size=3, activation = 'relu'),
    tf.keras.layers.AveragePooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    
    tf.keras.layers.Dense(CLASSES, activation='softmax') 
])

CLASSES = 2

def build_classifier(inputs):

    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 3))(inputs)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)
    
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    x = tf.keras.layers.Dense(CLASSES, activation='softmax')(x)

    return x