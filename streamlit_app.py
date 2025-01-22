import streamlit as st
import requests
from PIL import Image
import io

# Configure the app
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="centered"
)

# Title and description
st.title("Medical Image Analysis")
st.write("Upload an image and select the type of analysis you want to perform.")

# Disease selection
disease_type = st.selectbox(
    "Select Disease Type",
    ["Skin Cancer", "Pneumonia", "Eye", "Brain", "Heart"],
    help="Choose the type of disease you want to analyze"
)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Create predict button
    if st.button("Analyze Image"):
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Prepare the files for the API request
        files = {
            "file": ("image.jpg", img_byte_arr, "image/jpeg")
        }
        
        try:
            # Make API request based on disease type
            if disease_type:
                response = requests.post("http://localhost:8000/api/v1/predict", files=files)  # Updated endpoint
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("Analysis Complete!")
                st.json(result)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the server: {str(e)}")
            st.info("Make sure the FastAPI backend is running on http://localhost:8000")