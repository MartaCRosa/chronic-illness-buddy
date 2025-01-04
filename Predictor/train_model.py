import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import create_model

# Load and Preprocess Data
data = pd.read_csv("Predictor/Glucose_Level_Estimation.csv")

def preprocess_data(df):
    df = df.drop(columns=['NIR_Reading', 'HR_IR', 'SKIN_COLOR'])  # Remove columns with feautures that isn't data a patient could easily give
    df = df.dropna(subset=['WEIGHT', 'HEIGHT', 'DIABETIC'])  #Drop rows with missing features
    df['HEARTRATE'] = df['HEARTRATE'].fillna(df['HEARTRATE'].median())  # Use median of values missing
    return df

def remove_outliers(df, columns, threshold=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Keep data within bounds
    return df

# Apply preprocessing and outlier removal
data = preprocess_data(data)

# Specify numerical columns to remove outliers
numerical_columns = ['AGE', 'WEIGHT', 'HEARTRATE', 'HEIGHT', 'GLUCOSE_LEVEL']

# Remove outliers using the IQR method
data = remove_outliers(data, numerical_columns)

# Classify Glucose Levels
def classify_glucose(last_eaten, glucose_level):
    if last_eaten == -1:  # Fasting
        if glucose_level > 130:
            return "Hyperglycemia"
        elif 80 <= glucose_level <= 130:
            return "Normal"
        else:
            return "Hypoglycemia"
    elif last_eaten >= 0:
        if glucose_level > 180:
            return "Hyperglycemia"
        elif 80 <= glucose_level <= 180:
            return "Normal"
        else:
            return "Hypoglycemia"
    else:
        return "Unknown"

# Labeling for classification
data['CLASS'] = data.apply(lambda row: classify_glucose(row['LAST_EATEN'], row['GLUCOSE_LEVEL']), axis=1)
data = data[data['CLASS'] != 'Unknown']  # Remove rows with 'Unknown' classification
y = data['CLASS']
X = data.drop(columns=['GLUCOSE_LEVEL', 'CLASS'])

# Encode Categorical Columns and Target Variable
X = pd.get_dummies(X, columns=['GENDER', 'DIABETIC'], drop_first=True)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train/Test/Validation Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize and Train the Model
input_dim = X_train.shape[1]
output_dim = len(label_encoder.classes_)
model = create_model(input_dim=input_dim, output_dim=output_dim)

y_train_cat = to_categorical(y_train, num_classes=output_dim)
y_val_cat = to_categorical(y_val, num_classes=output_dim)
y_test_cat = to_categorical(y_test, num_classes=output_dim)

# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,          # Stop after x epochs with no improvement
    restore_best_weights=True  # Revert to the best model weights
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=400,
    batch_size=128,
    verbose=1,
    callbacks=[early_stopping]
)

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

results = model.evaluate(X_test, y_test_cat, verbose=0)
test_loss = results[0]
test_accuracy = results[1]
test_precision = results[2]
test_recall = results[3]
test_mse = results[4]

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test MSE: {test_mse}")

y_pred_class = np.argmax(model.predict(X_test), axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Save Model and Scaler
model.save("Predictor\Model\glucose_model.h5")
np.save("Predictor\Model\scaler.npy", scaler.mean_)
np.save("Predictor\Model\scaler_std.npy", scaler.scale_)
with open("Predictor\Model\label_encoder_classes.npy", "wb") as f:
    np.save(f, label_encoder.classes_)