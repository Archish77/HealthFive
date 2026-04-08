import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Data Load karein
df = pd.read_csv('Kidney_Disease.csv')

mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Physical_Activity_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Salt_Intake_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Protein_Intake': {'Normal': 0, 'High': 1}, # Data ke mutabik
    'Smoking': {'No': 0, 'Yes': 1},
    'Alcohol': {'No': 0, 'Yes': 1},
    'BP_Level': {'Normal': 0, 'High': 1},
    'Diabetes': {'No': 0, 'Yes': 1}
}

for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)

X = df.drop('Kidney_Disease', axis=1)
y = df['Kidney_Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

with open('Kidney_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")