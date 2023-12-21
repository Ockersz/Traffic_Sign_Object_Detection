from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = '/model.py'
model = load_model(model_path)

def process_image(file):
    img = Image.open(file.file)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_array = process_image(file)
        prediction = model.predict(img_array)
        return JSONResponse(content={"prediction": prediction.tolist()}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
