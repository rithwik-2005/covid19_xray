import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Simplified page config
st.set_page_config(page_title="COVID-19 Classifier", page_icon="🫁")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('covid19_xray_model.h5')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def preprocess_image(img):
    img = img.convert('RGB').resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def main():
    st.title("🫁 COVID-19 X-ray Classifier")
    st.warning("⚠️ For educational purposes only. Not for medical diagnosis.")
    
    model = load_model()
    if not model:
        return
    
    uploaded_file = st.file_uploader("Upload chest X-ray", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file and st.button("Analyze"):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray")
        
        with st.spinner("Processing..."):
            try:
                img_array = preprocess_image(img)
                prediction = model.predict(img_array)[0][0]
                covid_prob = round(prediction * 100, 2)
                normal_prob = round((1 - prediction) * 100, 2)
                
                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.metric("COVID-19 Probability", f"{covid_prob}%")
                col2.metric("Normal Probability", f"{normal_prob}%")
                
                if prediction > 0.5:
                    st.error("High COVID-19 risk - Please consult a doctor")
                else:
                    st.success("Low COVID-19 risk")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()