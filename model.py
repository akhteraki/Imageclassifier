import tensorflow as tf
import tensorflow_hub as hub

def load_model():
    IMAGE_SHAPE = (224, 224)
    classifier = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
    ])
    return classifier
