from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict")
def predict_species(input_data: IrisInput):
    return predict(input_data.dict())
