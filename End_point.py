from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from fastapi import Request

# Initialize FastAPI app
End_point = FastAPI()

# Load the model
model = tf.keras.models.load_model("cat_dog_classifier.keras")

# Set up Jinja2 template for HTML rendering, adjust directory path
templates = Jinja2Templates(directory=".")

# Prepare image function for prediction
def prepare_image(image: BytesIO, target_size=(150, 150)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    
    # Return the predicted class based on the model
    if prediction[0] > 0.5:
        return {"predicted_class": "Dog", "prediction_score": float(prediction[0])}
    else:
        return {"predicted_class": "Cat", "prediction_score": float(prediction[0])}

# Route to serve HTML form for image upload
@End_point.get("/", response_class=HTMLResponse)
async def form_post(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

# Route to handle image upload and classification
@End_point.post("/classification/")
async def classify_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_stream = BytesIO(image_bytes)
    result = prepare_image(image_stream)
    return result
