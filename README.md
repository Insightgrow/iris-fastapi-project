# Iris Flower Classification API using FastAPI

This is a simple and efficient **Machine Learning API** built using **FastAPI** to classify **Iris flower species** (`Setosa`, `Versicolor`, `Virginica`) based on user-provided measurements of sepal and petal length and width.

---

## Features

- **Predict Iris Species** using a trained RandomForestClassifier
- **FastAPI-based Web API** – blazing fast and lightweight
- **Machine Learning Integration** – Scikit-learn, joblib
- **JSON API Support** for integration with frontend apps or scripts


---

## Tech Stack

- **FastAPI** – Web framework
- **Scikit-learn** – For ML model training
- **Joblib** – Model serialization
- **Uvicorn** – ASGI server
- **Python**

---

## Project Structure
iris-fastapi-project/
│
├── app/
│ ├── main.py # FastAPI app code
│ ├── iris_model.pkl # Trained ML model
│ ├── label_encoder.pkl # Label encoder for species
├── README.md

