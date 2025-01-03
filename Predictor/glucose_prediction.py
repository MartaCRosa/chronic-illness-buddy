import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from model import create_model

data = pd.read_csv("Predictor\Glucose_Level_Estimation.csv")
#Visualization of data
#data.info()

# Check initial missing values
print("Initial missing values per column:")
print(data.isnull().sum())
print(data.shape)

# Pre-Processing
def preprocess_data(df):  
    df = df.drop(columns=['NIR_Reading', 'HR_IR','SKIN_COLOR'])  # Drop irrelevant columns
    #df = df[df['DIABETIC'] == 'Y']  # Filter for diabetic patients only    
    #df = df.drop(columns=['DIABETIC'])  # Not needed as a label
    df = df.dropna(subset=['WEIGHT', 'HEIGHT']) # Drop rows with missing values in critical columns   
    df['HEARTRATE'] = df['HEARTRATE'].fillna(df['HEARTRATE'].median()) # Fill missing values in numerical columns with median
    return df

data = preprocess_data(data)

print("\nMissing values:")
print(data.isnull().sum())
print(data.shape)

# Split Features and Target
X = data.drop(columns=['GLUCOSE_LEVEL'])
y = data['GLUCOSE_LEVEL']

# One-Hot Encode categorical columns
X = pd.get_dummies(X, columns=['GENDER', 'DIABETIC'], drop_first=True)

# Train/Test/Validation Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Train Model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Model Evaluation on Validation Set
y_val_pred = model.predict(X_val)
print("Validation MSE:", mean_squared_error(y_val, y_val_pred))

#fasting: 80 and 130 mg/dL
#2 hours after meals: <180 mg/dL

# 7. Classification of Predicted Glucose Levels
def classify_glucose_levels(predictions):
    categories = []
    count = 0
    for value in predictions:
        if value < 70:
            categories.append(0)  # Hypoglycemia
        elif 70 <= value <= 140:
            categories.append(1)  # Normal
        else:
            categories.append(2)  # Hyperglycemia
    return categories


y_test_pred = model.predict(X_test)
y_test_classes = classify_glucose_levels(y_test_pred)

# 8. Evaluation
actual_classes = classify_glucose_levels(y_test)
print("Classification Report:\n", classification_report(actual_classes, y_test_classes))
print("Confusion Matrix:\n", confusion_matrix(actual_classes, y_test_classes))

# 9. Save Model
#import joblib
#joblib.dump(model, "glucose_prediction_linear_model.pkl")

# Optional: Predict on a new example
# example_input = X_test.iloc[0:1]
# prediction = model.predict(example_input)
# print("Predicted Glucose Level:", prediction)
# print("Classification:", classify_glucose_levels(prediction))"""