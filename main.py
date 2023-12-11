from fastapi import FastAPI, HTTPException
import uvicorn

from pydantic import BaseModel, PrivateAttr, Field, PositiveFloat, computed_field
import joblib
import pandas as pd
import numpy as np
import sklearn

app = FastAPI()


class InputData(BaseModel):
    vendor_id :int
    passenger_count:int
    pickup_longitude: float
    pickup_latitude:float
    dropoff_longitude:float
    dropoff_latitude:float

    store_and_fwd_flag:str

# chargement du modele


model = joblib.load("models/tripduration.model")

@app.get("/")
def root():
    return {"message": "Hello!"}

@app.post("/predict/")
def get_model_predict(data: InputData):
    # Convertir les caractéristiques d'entrée en tableau NumPy
    features = np.array([[data.vendor_id, data.passenger_count,data.pickup_longitude,data.pickup_latitude,data.dropoff_longitude,data.dropoff_latitude,data.dropoff_datetime,data.store_and_fwd_flag]])

    # Effectuer la prédiction avec le modèle
    prediction_result = model.predict(features)

    #Convertir la prediction dans l'etat de base
    prediction_convertis = np.expm1(prediction_result[0])

    # Retourner le résultat de la prédiction
    return {"prediction": prediction_convertis}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
