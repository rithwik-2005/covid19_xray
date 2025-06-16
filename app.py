import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(page_title="COVID-19 X-ray Classifier", page_icon="🫁")

# Load model with caching
@st.cache_resource
def load_covid_model():
    return load_model('covid19_xray_model.h5')

def preprocess_image(input_image):
    img = input_image.convert('RGB').resize((256, 256))
    return np.expand_dims(image.img_to_array(img), axis=0) / 255.0

def predict_covid(input_image, model):
    img_array = preprocess_image(input_image)
    return model.predict(img_array)[0][0]

# Main app
def main():
    st.title("🫁 COVID-19 Chest X-ray Classifier")
    st.warning("⚠️ Medical Disclaimer: For educational purposes only. Not for actual diagnosis.")
    
    model = load_covid_model()
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("1. Upload a chest X-ray\n2. Click 'Analyze'\n3. View results")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded X-ray")
        
        if st.button("Analyze X-ray"):
            with st.spinner("Analyzing..."):
                prediction = predict_covid(image_pil, model)
                confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                result = "Positive" if prediction > 0.5 else "Negative"
                
                st.success(f"Result: {result} for COVID-19")
                st.write(f"Confidence: {confidence:.1f}%")
                st.progress(confidence / 100)
                
                if prediction > 0.5:
                    st.error("High likelihood of COVID-19. Please consult a doctor.")
                else:
                    st.info("Low likelihood, but consult a doctor for clarity.")

if __name__ == "__main__":
    main()