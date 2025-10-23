import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
from dataCleaning.preprocessing import clean_data
import pandas as pd
import uvicorn
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

load_dotenv()

app = FASTAPI(title = "Buffer Stock Prediction", version = '1.0.0')
#Tells FASTAPI to create RESTFUL API called "Buffer Stock Prediction"

class PredictionRequest(BaseModel):
    """The Class function inherits the attributes of Pydantic BaseModel #Pydantic defines the input data structure
     expected by the model and validates it automatically"""
     records: List[Dict[str, Any]] = Field(
        ...,
        example=[
        {
            'Wastage_Units': 800,
            'Product_Name': 'Whole Wheat Bread 800g',
            'Product_Category': 'Bakery',
            'Shelf_Life_Days':4,
            'Price': 3.5,
            'Cold_Storage_Capacity': 500,
            'Store_Size': 12000,
            'Rainfall': 20.5
            'Avg_Temperature':22.3
            'Region': 'London'

        }
        ]
     )


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "Model")
MODEL_DIR_ = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR_, "rf_best_regressor.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR_, "feature_schema.json")

_model = None
_schema_cols = None

def load_assets():
    global _model, _schema_cols
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
        
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    _schema_cols = schema["feature_columns"]

load_assets()

@app.on_event("startup")
def _on_startup():
    global _model, _schema_cols
    if _model is None or _schema_cols is None:
        load_assets()

@app.post('/predict')
def predict(req: PredictionRequest):