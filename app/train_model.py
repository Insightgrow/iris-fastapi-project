import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("iris.csv") 

X = df.drop("species", axis=1)
y = df["species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, "iris_model.pkl")

print("Model trained and saved successfully.")
