

import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

st.title('Military Vehicle Detection')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Prepare the file to send to FastAPI
    files = {'files': (uploaded_file.name, uploaded_file, uploaded_file.type)}

    # Post the file to the FastAPI endpoint
    response = requests.post('http://localhost:8000/upload/', files=files)

    # Display the response
    if response.status_code == 200:
        response_data = response.json()
        st.write(response_data)

        # Load the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Draw a rectangle on the image
        # Assuming box = [x1, y1, x2, y2]
        box = [50, 50, 200, 200]  # Example box coordinates
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)

        # Display the image with the rectangle
        st.image(image, caption='Detected Object.', use_column_width=True)
    else:
        st.error("Failed to process the image.")

