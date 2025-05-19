import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Dynamically get the full path to the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../dataset/Iris.csv")

# Load dataset
df = pd.read_csv(data_path)

# Drop ID column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Encode target
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Split
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model & encoder
joblib.dump(model, os.path.join(current_dir, "model.joblib"))
joblib.dump(le, os.path.join(current_dir, "label_encoder.joblib"))
print("Model training complete and saved successfully.")