import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris ML API")
model = joblib.load("model.joblib")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[features.sepal_length, features.sepal_width,
             features.petal_length, features.petal_width]]
    pred = int(model.predict(data)[0])
    return {"prediction": pred, "class_name": CLASS_NAMES[pred]}