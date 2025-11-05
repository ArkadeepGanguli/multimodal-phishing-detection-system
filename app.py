import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv not available in this environment; provide a no-op fallback
    def load_dotenv():
        return None

# ----------------------------
# 🔹 Load URL-based Model (.pkl)
# ----------------------------
url_model = pickle.load(open('phishing.pkl', 'rb'))

# ----------------------------
# 🔹 Define the SAME focal loss function used during training
# ----------------------------
# ✅ Register BOTH the outer and inner functions
@tf.keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    focal = alpha * (1 - bce_exp) ** gamma * bce
    return focal

# ✅ If you used a wrapper during training, keep this for compatibility
@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        return focal_loss_fixed(y_true, y_pred, gamma=gamma, alpha=alpha)
    return loss

# ✅ Load model with both registered
img_model = keras.models.load_model(
    'phishing_screenshot_mobilenetv2_focal.keras',
    custom_objects={
        'focal_loss': focal_loss,
        'focal_loss_fixed': focal_loss_fixed
    },
    compile=False
)

# ----------------------------
# 🔹 Streamlit UI
# ----------------------------
st.title("🛡️ Multimodal Phishing Detector")

# Load .env if present
load_dotenv()
API_KEY = os.getenv('SCREENSHOT_API_KEY')  # fallback to existing key
API_URL = "https://shot.screenshotapi.net/screenshot"

# ----------------------------
# 🔹 Function to Fetch Screenshot
# ----------------------------
def get_screenshot(url):

    params = {
        "token": API_KEY,
        "url": url,
        "full_page": "true",
        "output": "image",
        "file_type": "png"
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        st.error(f"Screenshot API error: {response.status_code}")
        return None

def clean_url(input_url: str) -> str:
    if not input_url:
        return ""
    url = input_url.strip().lower()
    prefixes = ["https://www.", "http://www.", "https://", "http://", "www."]
    for prefix in prefixes:
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    url = url.rstrip('/')
    return url

# ----------------------------
# 🔹 Input and Prediction
# ----------------------------
url = st.text_input("Enter Website URL:")

if st.button("Predict") and url.strip():
    st.info("🔍 Analyzing your site across URL & visual features...")

    # 1️⃣ URL Model Prediction
    cleaned_url = clean_url(url)
    url_pred = url_model.predict_proba([cleaned_url])[0][1]
    # st.write(f"🌐 URL Model Phishing Probability: **{(1 - url_pred):.2f}**")

    # 2️⃣ Screenshot Capture
    image_file = get_screenshot(url)

    # 3️⃣ Image Model Prediction
    if image_file is not None:
        st.image(image_file, caption="Website Screenshot", use_container_width=True)

        img = image_file.convert('RGB').resize((128, 128))
        img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        img_pred = float(img_model.predict(img_arr)[0][0])
        # st.write(f"🖼️ Image Model Phishing Probability: **{img_pred:.2f}**")
    else:
        img_pred = 0.5  # Neutral fallback if screenshot failed
        st.warning("⚠️ Could not retrieve screenshot; using neutral score for image model.")

    # 4️⃣ Fused Decision (Weighted)
    final_score = (0.6 * (1 - url_pred)) + (0.4 * img_pred)
    # Compute confidence percentage and format the result for both branches
    confidence_pct = final_score * 100
    if final_score > 0.5:
        label = "🚫 **Phishing (Malicious)**"
    else:
        label = "✅ **Legitimate (Safe)**"
    result = f"{label} — {confidence_pct:.1f}% Confidence"

    st.success(f"Final Decision: {result}")
    st.progress(final_score)
