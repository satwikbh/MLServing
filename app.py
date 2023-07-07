import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI

app = FastAPI()

model_path = f"{os.getcwd()}/MnistCNNModel.h5"
model = tf.keras.models.load_model("path/to/your/pretrained/model")


@app.post("/predict")
def predict(image: np.ndarray):
    # Preprocess the image if needed
    # Perform prediction using the loaded model
    predictions = model.predict(image)
    # Process the predictions if required
    return {"predictions": predictions.tolist()}
