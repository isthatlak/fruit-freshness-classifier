# streamlit_app.py - Fruit Freshness Classification Web App (Hugging Face)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

# Configure page
st.set_page_config(
    page_title="🍎 Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide"
)

# ---------------------------
# Download & Load Model from Hugging Face
# ---------------------------
@st.cache_resource
def download_model():
    """
    Download the TensorFlow .h5 model from Hugging Face
    and load it into memory.
    """
    model_path = hf_hub_download(
        repo_id="isthatlak/fruit_freshness_model",
        filename="fruit_freshness_model.h5",
        repo_type="model",
        use_auth_token=False
    )
    return tf.keras.models.load_model(model_path)

# ---------------------------
# Define Classes & Colors
# ---------------------------
CLASSES = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

FRESHNESS_COLORS = {
    'fresh': '#28a745',  # Green
    'rotten': '#dc3545'  # Red
}

def get_freshness_status(class_name):
    """Determine if fruit is fresh or rotten"""
    return 'fresh' if 'fresh' in class_name.lower() else 'rotten'

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(image):
    """Preprocess image for model prediction"""
    img_resized = image.convert("RGB").resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# ---------------------------
# Prediction Chart
# ---------------------------
def create_prediction_chart(predictions, classes):
    prob_data = []
    for cls, prob in zip(classes, predictions):
        freshness = get_freshness_status(cls)
        prob_data.append({'Class': cls, 'Probability': prob, 'Freshness': freshness})
    prob_data = sorted(prob_data, key=lambda x: x['Probability'], reverse=True)

    fig = px.bar(
        prob_data,
        x='Probability',
        y='Class',
        orientation='h',
        color='Freshness',
        color_discrete_map=FRESHNESS_COLORS,
        title='Prediction Probabilities'
    )
    fig.update_layout(
        xaxis_title='Probability',
        yaxis_title='Class',
        showlegend=True,
        height=400
    )
    return fig

# ---------------------------
# Confidence Gauge
# ---------------------------
def show_confidence_gauge(confidence, freshness):
    color = "green" if freshness == 'fresh' else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#f2f2f2"},
                {'range': [50, 100], 'color': "#d9d9d9"}
            ],
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown("""
    # 🍎 Fruit Freshness Classifier
    Upload an image of a fruit to determine if it's **fresh** or **rotten**!
    
    **Supported fruits:** Apples, Bananas, Oranges
    """)

    # Load model
    with st.spinner("Loading AI model..."):
        model = download_model()

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Model:** MobileNetV2 Transfer Learning
        **Classes:** 6 categories
        - Fresh: Apples, Bananas, Oranges  
        - Rotten: Apples, Bananas, Oranges
        **Input:** 224x224 RGB images
        """)
        st.header("🎯 How it works")
        st.write("""
        1. **Upload** an image of a fruit
        2. **AI analyzes** the image features
        3. **Predicts** freshness with confidence
        4. **Visualizes** prediction probabilities and confidence
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fruit image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of an apple, banana, or orange"
        )

        # Camera input
        camera_image = None
        if st.button("📷 Take a photo with your camera"):
            camera_image = st.camera_input("Take a photo")

        image_source = camera_image if camera_image else uploaded_file

        if image_source is not None:
            image = Image.open(image_source)
            st.image(image, caption='Selected Image', width=300)
            if st.button("🔍 Analyze Fruit", type="primary"):
                with st.spinner("Analyzing image..."):
                    img_array = preprocess_image(image)
                    predictions = model.predict(img_array)[0]
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = CLASSES[predicted_class_idx]
                    confidence = float(predictions[predicted_class_idx])
                    freshness = get_freshness_status(predicted_class)
                    
                    st.session_state.predictions = predictions
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.freshness = freshness
                    st.session_state.image_array = img_array

    with col2:
        st.header("🎯 Analysis Results")
        if hasattr(st.session_state, 'predicted_class'):
            freshness = st.session_state.freshness
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence

            if freshness == 'fresh':
                st.success(f"✅ **{predicted_class.upper()}**")
            else:
                st.error(f"❌ **{predicted_class.upper()}**")

            show_confidence_gauge(confidence, freshness)
            st.subheader("📊 Detailed Predictions")
            chart = create_prediction_chart(st.session_state.predictions, CLASSES)
            st.plotly_chart(chart, use_container_width=True)

            st.subheader("💡 Recommendations")
            if freshness == 'fresh':
                st.markdown("""
                🟢 **This fruit appears fresh!**
                - Safe to eat
                - Store properly
                - Consume within recommended timeframe
                """)
            else:
                st.markdown("""
                🔴 **This fruit appears rotten!**
                - Do not consume
                - Dispose properly
                - Check other fruits for contamination
                """)
        else:
            st.info("👆 Upload an image and click 'Analyze Fruit' to see results")

    st.markdown("---")
    st.subheader("ℹ️ About This App")
    st.markdown("""
    - Deep Learning: TensorFlow/Keras  
    - Model: MobileNetV2 + Transfer Learning  
    - Frontend: Streamlit  
    - Visualizations: Top predictions & confidence gauge  
    - Works best with apples, bananas, oranges  
    """)

if __name__ == "__main__":
    main()