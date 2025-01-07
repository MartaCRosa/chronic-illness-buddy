import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load Trained Model and Preprocessing Tools
model = load_model("Predictor\Trained_Model\glucose_model.h5")
scaler_mean = np.load("Predictor\Trained_Model\scaler.npy")
scaler_std = np.load("Predictor\Trained_Model\scaler_std.npy")
label_encoder_classes = np.load("Predictor\Trained_Model\label_encoder_classes.npy", allow_pickle=True)

# Feature Preprocessing Function
def preprocess_features(age, gender, weight, heartrate, height, last_eaten, diabetic):
    # Convert categorical features to numeric
    gender_encoded = 1 if gender == "M" else 0
    diabetic_encoded = 1 if diabetic == "Y" else 0

    # Combine features into a single array
    features = np.array([age, weight, heartrate, height, last_eaten, gender_encoded, diabetic_encoded])
    
    # Normalize numerical features using the saved scaler
    normalized_features = (features - scaler_mean) / scaler_std
    return normalized_features

# Classification Function
def classify_glucose(age, gender, weight, heartrate, height, last_eaten, diabetic):
    # Preprocess input features
    processed_features = preprocess_features(age, gender, weight, heartrate, height, last_eaten, diabetic)
    
    # Predict class probabilities
    predictions = model.predict(np.array([processed_features]))
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map to class label
    print(label_encoder_classes[predicted_class])
    return label_encoder_classes[predicted_class]

# Example Usage
if __name__ == "__main__":
    # Input example features
    age = 55
    gender = "F"
    weight = 70
    heartrate = 65
    height = 5.4
    last_eaten = 4
    diabetic = "Y"
    
    # Get prediction
    glucose_class = classify_glucose(age, gender, weight, heartrate, height, last_eaten, diabetic)
    print(f"Predicted Glucose Level: {glucose_class}")
