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

app = FastAPI(title = "Buffer Stock Prediction", version = '1.0.0')
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
            'Rainfall': 20.5,
            'Avg_Temperature':22.3,
            'Region': 'London'

        }
        ]
     )

@app.post('/predict')
def predict(req: PredictionRequest):
    try:
        df = pd.DataFrame(req.records)
        cleaned_df = clean_data(df)

        MODEL_PATH = os.path.join(os.path.dirname(__file__), '..','Model','artifacts','rf_best_regressor.pkl')
        with open(MODEL_PATH, 'rb') as f:
            model =pickle.load(f)
        
        pred = model.predict(cleaned_df)
 
        return {'Buffer Stock': predict.tolist()}
    except Exception as e:
        print(f' Error during prediction: {e}')
        raise HTTPException(status_code = 500, detail = f'Data Cleaning Error: {str(e)}')
if __name__ == '__main__':
    uvicorn.run('app:app', host = 127.0.0.1, port = 8080, reload = True)