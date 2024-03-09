from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image, ImageDraw
import io
from typing import List
import uvicorn
import sqlite3
from keras.models import load_model

model_path = 'C:\\military_vehicle_detection\\models\\military_vehicle_model.h5'
weights_path = 'C:\\military_vehicle_detection\\models\\military_vehicle_model_weights.h5'

model = load_model(model_path)
model.load_weights(weights_path)

app = FastAPI()

def insert_record(filename, details):
    conn = sqlite3.connect('C:\\military_vehicle_detection\\database\\file_uploads.db')
    c = conn.cursor()
    c.execute("INSERT INTO uploads (filename, details) VALUES (?, ?)", (filename, details))
    conn.commit()
    conn.close()

@app.post("/upload/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if the image is RGBA or grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((150, 150))  # Resize the image to match the model's expected input
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        box = [50, 50, 200, 200]  # Example box coordinates

        # Draw the rectangle on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)

        # Save or display the image as needed
        # For example, save the modified image
        image.save(f"detected_{file.filename}")

        insert_record(file.filename, str(predicted_class))



        insert_record(file.filename, str(predicted_class))
        results.append({"filename": file.filename, "details": str(predicted_class)})

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
