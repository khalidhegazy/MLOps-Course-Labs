import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

def predict(data):
    features = np.array([[
        data["SepalLengthCm"],
        data["SepalWidthCm"],
        data["PetalLengthCm"],
        data["PetalWidthCm"]
    ]])
    
    pred_class = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    
    return {"prediction": pred_label}
