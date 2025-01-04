from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import create_model

# Load and Preprocess Data
data = pd.read_csv("Predictor\Glucose_Level_Estimation.csv")

def preprocess_data(df):
    df = df.drop(columns=['NIR_Reading', 'HR_IR', 'SKIN_COLOR'])
    df = df.dropna(subset=['WEIGHT', 'HEIGHT'])
    df['HEARTRATE'] = df['HEARTRATE'].fillna(df['HEARTRATE'].median())
    return df

data = preprocess_data(data)

# Split Features and Target
X = data.drop(columns=['GLUCOSE_LEVEL'])
y = data['GLUCOSE_LEVEL']

# One-Hot Encode categorical columns
X = pd.get_dummies(X, columns=['GENDER', 'DIABETIC'], drop_first=True)

# Train/Test/Validation Split
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Before normalization
data.drop(columns=['GLUCOSE_LEVEL']).hist(figsize=(15, 10))
plt.suptitle('Feature Distributions Before Normalization')
plt.show()

# After normalization
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_train_df.hist(figsize=(15, 10))
plt.suptitle('Feature Distributions After Normalization')
plt.show()

# Initialize and Train the Model
input_dim = X_train.shape[1]
model = create_model(input_dim=input_dim)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=400,
    batch_size=128,
    verbose=1
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test.round(), y_pred.round())  # For comparison with true rounded values
precision = precision_score(y_test.round(), y_pred.round(), average='weighted', zero_division=0)
recall = recall_score(y_test.round(), y_pred.round(), average='weighted', zero_division=0)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Glucose Levels')
plt.ylabel('Predicted Glucose Levels')
plt.title('Predicted vs Actual Glucose Levels')
plt.show()

print("\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Print Predicted Glucose Levels
print("\nPredicted Glucose Levels:")
print(y_pred.flatten())

residuals = y_test - y_pred.flatten()
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()

# Optional: Save the trained model
#model.save("glucose_prediction_model.h5")
