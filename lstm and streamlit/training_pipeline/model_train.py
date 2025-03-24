import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.paths_config import *
from src.logger import get_logger
import dagshub
import pandas as pd
import numpy as np
import traceback
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
from statistics import mean
import os
import pickle
from sklearn.metrics import root_mean_squared_error


logger= get_logger(__name__)

logger.info(f"The path to processed train path: {PROCESSED_TRAIN_DATA_PATH}")
logger.info(f"The path to model dir: {MODEL_DIR}")

load_dotenv(override=True)

dagshub.init(repo_owner= os.getenv("MLFLOW_USERNAME"), repo_name= "WeatherProj", mlflow=True)
tracking_uri= os.getenv("MLFLOW_TRAKCING_URI")

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
    
class OptunaTuning:
    def __init__(self, train_dataloader, val_dataloader, device, scaler):
        self.train_dataloader= train_dataloader
        self.val_dataloader= val_dataloader
        self.device= device
        self.scaler= scaler

    def objective_model(self, trial):
        hidden_size= trial.suggest_int("hidden_size", 50, 250, step= 50)
        learning_rate= trial.suggest_float("learning_rate", 1e-5, 1e-2)
        dropout= trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay= trial.suggest_float("weight_decay", 1e-5, 1e-4)
        n_layers= trial.suggest_int("n_layers", 2, 5)
        n_epochs= trial.suggest_int("n_epochs", 10, 50)

        model= LSTMModel(4, hidden_size, n_layers, dropout).to(self.device)
        criterion= torch.nn.MSELoss().to(self.device)
        optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate)

        test_loss=[]
        for _ in range(n_epochs):
            model.train()

            for x, y in self.train_dataloader:
                x, y= x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output= model(x)
                loss= torch.sqrt(criterion(output.view(-1), y))
                loss.backward()
                optimizer.step()

                

        model.eval()
        
        with torch.no_grad():
            for x, y in self.val_dataloader:
                x, y= x.to(self.device), y.to(self.device)

                output= model(x)
                loss= torch.sqrt(criterion(output.view(-1), y))
                test_loss.append(loss.item())
        
        # eval stage

        return mean(test_loss)
    
    def objective(self, parent_run):
        with mlflow.start_run(run_id= parent_run.info.run_id, nested=True):
            study= optuna.create_study(direction= "minimize")
            study.optimize(self.objective_model, n_trials=25, n_jobs=4)
        return study

    
class ModelTrain:
    def __init__(self, train_path, test_path, seq):
        self.train_path= train_path
        self.test_path= test_path
        self.seq= seq

    def make_sequence(self, df, seq):
        x, target= [], []

        for i in range(len(df)- seq-1):
            s= df.iloc[i:i+seq]
            t= df.iloc[i+seq, 0]

            x.append(s)
            target.append(t)

        return np.array(x), np.array(target)
    

    def train(self):
        try:

            df_train= pd.read_csv(self.train_path, index_col="date", parse_dates=True)
            df_test= pd.read_csv(self.test_path, index_col="date", parse_dates=True)

            df_train= df_train.loc[:'2016-06', :]
            df_val= df_train.loc['2016-06': ,:]

            scaler= StandardScaler()

            df_train= pd.DataFrame(scaler.fit_transform(df_train), columns= scaler.get_feature_names_out())
            df_test= pd.DataFrame(scaler.transform(df_val), columns=scaler.get_feature_names_out())

            x_train, y_train= self.make_sequence(df_train, self.seq)
            x_val, y_val= self.make_sequence(df_val, self.seq)

            x_train.reshape(x_train.shape[0], self.seq, 4)
            x_val.reshape(x_val.shape[0], self.seq, 4)

            x_train= torch.tensor(x_train, dtype= torch.float32)
            y_train= torch.tensor(y_train, dtype= torch.float32)
            x_val= torch.tensor(x_val, dtype= torch.float32)
            y_val= torch.tensor(y_val, dtype= torch.float32)

            train_dataset= TensorDataset(x_train,  y_train)
            val_dataset= TensorDataset(x_val, y_val)

            train_dataloader= DataLoader(train_dataset, batch_size= 32, shuffle=True)
            val_dataloader= DataLoader(val_dataset, batch_size=32, shuffle=False)       

            device= "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Parameter Tuning started")

            with mlflow.start_run() as parent_run:
                optuna_tuning= OptunaTuning(train_dataloader, val_dataloader, device, scaler)
                study= optuna_tuning.objective(parent_run)

            logger.info("Parameter tuning completed successfully...")

            logger.info(f"Best params are: {study.best_params}")

            mlflow.log_metrics(study.best_params)

            best_params= study.best_params
            best_model= LSTMModel(
                4,
                best_params["hidden_size"],
                best_params["n_layers"],
                best_params["dropout"]
            ).to(device)

            optimizer= torch.optim.Adam(best_model.parameters(), lr= best_params["learning_rate"])
            criterion= torch.nn.MSELoss().to(device)

            logger.info("Best model training starting...")

            for epoch in range(best_params["n_epochs"]):
                train_loss= 0
                best_model.train()

                for x, y in train_dataloader:
                    x, y= x.to(device), y.to(device)

                    optimizer.zero_grad()
                    output= best_model(x)

                    loss= torch.sqrt(criterion(output.view(-1), y))
                    loss.backward()
                    optimizer.step()
                    train_loss+= loss.item()
                
                best_model.eval()
                val_loss=0
                with torch.no_grad():
                    for x, y in val_dataloader:
                        x, y= x.to(device), y.to(device)

                        output= best_model(x)
                        loss= torch.sqrt(criterion(output.view(-1), y))
                        val_loss+= loss.item()

                
                logger.info(f"Epoch: {epoch+1}/{best_params['n_epochs']} TrainLoss: {train_loss/len(train_dataloader)} ValLoss: {val_loss/len(val_dataloader)}")



            logger.info("Best model training completed...")

            lstm_model= mlflow.pytorch.log_model(
                best_model,
                "pytorch_lstm_model",
                registered_model_name= "pytorch_lstm_model"
            )

            sklearn_model= mlflow.sklearn.log_model(
                    scaler,
                    "sklearn_scaler",
                    
            )

            logger.info("Training completed")

            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(best_model.state_dict(), PyTorch_MODEL_OUTPUT_PATH)
            joblib.dump(scaler, SKLEARN_MODEL_OUTPUT_PATH)
            
            with open(PyTorch_Params, "wb") as f:
                pickle.dump(best_params, f, protocol= pickle.HIGHEST_PROTOCOL)

            #model= torch.load(PyTorch_MODEL_OUTPUT_PATH)
            #scaler= joblib.load(SKLEARN_MODEL_OUTPUT_PATH)

            df_test= pd.DataFrame(scaler.transform(df_test), columns= scaler.get_feature_names_out())
            x_test, y_test= self.make_sequence(df_test, self.seq)
            x_test.reshape(x_test.shape[0], self.seq, 4)
            x_test= torch.tensor(x_test, dtype= torch.float32)
            y_test= torch.tensor(y_test, dtype= torch.float32)

            test_dataset= TensorDataset(x_test, y_test)
            
            test_dataloader= DataLoader(test_dataset, batch_size= 32, shuffle=False)

            best_model.eval()
            test_loss= []
            with torch.no_grad():
                for x,y in test_dataloader:
                    x= x.to(device)

                    output= best_model(x).detach().cpu().numpy()
                    output= np.tile(output, (1, 4))
                    output= scaler.inverse_transform(output)
                    temperature_out= output[:, 0]

                    test_loss.append(root_mean_squared_error(y, temperature_out))
                
                logger.info(f"Testing loss: {mean(test_loss)}")

            logger.info("All the model training and evaluation completed successfully...")
        
        except Exception:
            #logger.error("Error raised...")
            logger.error(traceback.format_exc())


if __name__=="__main__":
    model_train= ModelTrain(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, 7)
    model_train.train()




