import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

# Pydantic model for input validation
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

# Lifespan handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and preprocessor during app startup
    app.state.model = joblib.load("model.pkl")
    app.state.preprocessor = joblib.load("preprocessor.pkl")
    yield

app = FastAPI(lifespan=lifespan)
instrumentator = Instrumentator().instrument(app).expose(app)
@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])  # <-- Fix is here

    # Retrieve model and preprocessor from app state
    model = app.state.model
    preprocessor = app.state.preprocessor

    # Transform input and make prediction
    X_input = preprocessor.transform(input_df)
    churn_proba = model.predict_proba(X_input)[0][1]
    churn_pred = int(model.predict(X_input)[0])

    return {
        "churn_probability": churn_proba,
        "churn_prediction": churn_pred
    }
