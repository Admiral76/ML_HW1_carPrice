from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import numpy as np
import pandas as pd
import re



app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]




def preprocessing(data: List[Item]) -> pd.DataFrame:
    with open('pkl_params/medians.pickle', 'rb') as md:
        medians = pickle.load(md)
    

    df = pd.DataFrame([d.dict() for d in data])
   
    for col in ('mileage', 'engine', 'max_power'):
        df[col] = df[col].apply(lambda x: re.search(r'[0-9]*[.,]?[0-9]+', str(x)).group() 
                                if re.search(r'[0-9]*[.,]?[0-9]+', str(x)) != None 
                                else np.nan).astype(float)
    
    for col in medians.keys():
        df[col] = df[col].fillna(medians[col])

    return df


def predictor(items: List[Item]) -> List[float]:
    with open('pkl_params/ridge_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('pkl_params/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('pkl_params/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    data = preprocessing(items)
    
    num_columns = scaler.feature_names_in_
    cat_columns = encoder.feature_names_in_

    data = pd.concat(
        [
            pd.DataFrame(scaler.transform(data[num_columns]), columns=num_columns),
            pd.DataFrame(encoder.transform(data[cat_columns]).toarray(), columns=encoder.get_feature_names_out())
        ], axis=1
    )
    
    return list(model.predict(data))


@app.get('/')
def root() -> str:
    return 'Предсказание стоимости авто'

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return predictor([item])[0]

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return predictor(items)






