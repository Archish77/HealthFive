import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('brain_stroke.csv')

mappings = {
    'gender': {'Male': 1, 'Female': 0},
    'ever_married': {'Yes': 1, 'No': 0},
    'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3},
    'Residence_type': {'Urban': 1, 'Rural': 0},
    'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
}

for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)


X = df.drop('stroke', axis=1)
y = df['stroke']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as stroke_model.pkl")