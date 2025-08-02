import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation, Dropout
from PIL import Image
from keras.utils import register_keras_serializable

# Register swish activation (used in EfficientNet)
def swish(x):
    return tf.nn.swish(x)

get_custom_objects().update({'swish': Activation(swish)})

# Register FixedDropout (custom layer used in model)
@register_keras_serializable(package="efficientnet.model")
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

    def call(self, inputs, training=None):
        return super(FixedDropout, self).call(inputs, training=training)

# Load model with custom objects
model = load_model(
    "deepfake_efficientnet_model.keras",
    custom_objects={'swish': swish, 'FixedDropout': FixedDropout}
)

# Streamlit UI
st.title("Deepfake Detection App")
st.write("Upload a face image to check if it's real or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)


    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Result
    if prediction > 0.5:
        st.error(f"Fake Face Detected! Confidence: {prediction:.2f}")
    else:
        st.success(f"Real Face Detected! Confidence: {1 - prediction:.2f}")

