from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# ✅ Define the full 4-feature input model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ✅ Load model and label encoder from the 'app/' folder
model_path = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

model = joblib.load(model_path)
le = joblib.load(encoder_path)

# ✅ Prediction endpoint using all 4 features
@app.post("/predict")
def predict_species(features: IrisFeatures):
    try:
        data = np.array([[features.sepal_length, features.sepal_width,
                          features.petal_length, features.petal_width]])
        prediction = model.predict(data)
        species = le.inverse_transform(prediction)[0]
        return {"predicted_species": species}
    except Exception as e:
        return {"detail": f"Prediction error: {str(e)}"}
