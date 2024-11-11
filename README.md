# Image Classification with FastAPI

This FastAPI app classifies images as "Cat" or "Dog" using a pre-trained TensorFlow model.

## Setup:

1. Install dependencies: pip install tensorflow fastapi uvicorn pillow jinja2
2. Ensure the directory structure includes:
* app.py
* cat_dog_classifier.keras
* upload_form.html
3. Run the app: uvicorn app:End_point --reload
4. Access the app at http://127.0.0.1:8000/ to upload an image.
## How It Works:

1. Upload an image.
2. The model predicts whether it's a "Cat" or "Dog".
3. The result is shown with the prediction score.