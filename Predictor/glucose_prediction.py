import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#import seaborn as sns


df = pd.read_csv("Predictor\Glucose_Level_Estimation.csv")
#Visualization of data
#df.info()

# Check initial missing values
print("Initial missing values per column:")
print(df.isnull().sum())

""""
# 2. Create Target Column
def create_glucose_category(df):
    conditions = [
        df['GLUCOSE_LEVEL'] < 70,
        (df['GLUCOSE_LEVEL'] >= 70) & (df['GLUCOSE_LEVEL'] <= 140),
        df['GLUCOSE_LEVEL'] > 140
    ]
    categories = [0, 1, 2]  # 0: Hypoglycemia, 1: Normal, 2: Hyperglycemia
    df['GLUCOSE_CATEGORY'] = np.select(conditions, categories)
    return df

data = create_glucose_category(data)

# 3. Preprocess Dataset
def preprocess_data(df):
    # Drop irrelevant features (adjust based on your dataset)
    X = df.drop(columns=['GLUCOSE_LEVEL', 'GLUCOSE_CATEGORY'])
    y = df['GLUCOSE_CATEGORY']

    # One-Hot Encode categorical columns
    X = pd.get_dummies(X, columns=['GENDER', 'SKIN_COLOR', 'DIABETIC'], drop_first=True)

    return X, y

X, y = preprocess_data(data)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
importance = clf.feature_importances_
#sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.show()

# 7. Save Model
import joblib
joblib.dump(clf, "glucose_prediction_model.pkl")

# 8. Load Model for Prediction (Optional)
# loaded_model = joblib.load("glucose_prediction_model.pkl")
# example_input = X_test.iloc[0:1]
# prediction = loaded_model.predict(example_input)
# print("Prediction:", prediction)
"""
