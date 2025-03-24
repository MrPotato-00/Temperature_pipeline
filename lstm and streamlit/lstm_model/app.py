import pickle
import sys
import os
import traceback
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from flask import Flask, request, jsonify # type: ignore
#from src.logger import get_logger
from paths_config import *
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import psycopg2
#from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

#load_dotenv(override=True)

#logger= get_logger(__name__)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm= torch.nn.LSTM(
            input_size, 
            hidden_size,
            n_layers,
            dropout= dropout,
            batch_first=True
        )
        self.n_layers= n_layers
        self.fc= torch.nn.Linear(hidden_size, 1)
        self.hidden_size= hidden_size

    def forward(self, x):

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        x, _= self.lstm(x, (h0, c0))
        x= self.fc(x[:, -1, :])
        return x

with open(PyTorch_Params, "rb") as f:
    best_params= pickle.load(f)



model= LSTMModel(4, best_params["hidden_size"], best_params["n_layers"], best_params["dropout"])

model.load_state_dict(torch.load(PyTorch_MODEL_OUTPUT_PATH, map_location=torch.device("cpu")))
model.to("cpu")
scaler= joblib.load(SKLEARN_MODEL_OUTPUT_PATH)

app= Flask(__name__)

POSTGRES_URL= URL.create(
    drivername="postgresql",
    username= os.environ.get("username"),
    password= os.environ.get("password"),
    host= os.environ.get("host"),
    port= "5432",
    database= os.environ.get("database")
)

@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome Home</h1>"

def get_last_7days():
    engine= create_engine(POSTGRES_URL)
    
    query= "select * from weather_data order by date desc limit 7;"
    df= pd.read_sql(query ,engine)
    #print(df.head())
    df["date"]= pd.to_datetime(df["date"])
    
    df.set_index("date", inplace=True)
    df.rename(columns= {"pressure": "meanpressure"}, inplace=True)
    #print(df.index)
    return df
    


@app.route("/predict", methods=["GET"])
def predict():

    
    past_data= get_last_7days()

    df= pd.DataFrame(scaler.transform(past_data), columns=scaler.get_feature_names_out())
    #data= TensorDataset(df).unsqueeze(0)
    df.index= past_data.index
    #print(df.index)
    data= torch.tensor(df.values, dtype= torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output= model(data).numpy()
    output= np.tile(output, (1, 4))
    output= scaler.inverse_transform(output)
    output= output[0, 0]

    df.info()
    response= {
        "prediction_temperature": round(output.item(), 1), 
        "past_7days_temperature": past_data["temp"].tolist(),
        "dates": df.index.strftime("%Y-%m-%d").tolist()
    }
    #print(response)
    return jsonify(response)

if __name__=="__main__":
    app.run()