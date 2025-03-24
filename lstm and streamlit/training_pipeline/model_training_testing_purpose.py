from config.paths_config import *
from src.logger import get_logger
import dagshub
import pandas as pd
import numpy as np
import tensorflow as tf
import traceback
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlflow.models import infer_signature

logger= get_logger(__name__)

load_dotenv(override=True)

dagshub.init(repo_owner= os.getenv("MLFLOW_USERNAME"), repo_name= "WeatherProj", mlflow=True)
tracking_uri= os.getenv("MLFLOW_TRAKCING_URI")



class OptunaTuning:
    def __init__(self, x_train, y_train ,x_val, y_val):
        self.x_train= x_train
        self.y_train= y_train
        self.x_val= x_val
        self.y_val= y_val

    def objective_model(self, trial):
        #n_layers= trial.suggest_int("num_layers", 1, 5)
        n_nodes= trial.suggest_int("n_nodes", 32, 64)
        learning_rate= trial.suggest_float("learning_rate", 1e-5, 1e-2)
        dropout= trial.suggest_float("dropout", 0.1, 0.5)
        l2_regularizer= trial.suggest_float("l2_regularizer", 1e-5, 1e-2)

        model= tf.keras.models.Sequential()

        model.add(tf.keras.layers.LSTM(n_nodes, activation="relu",input_shape= (7, 4), kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer), return_sequences= True))
        model.add(tf.keras.layers.LSTM(n_nodes, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer), return_sequences= True))
        model.add(tf.keras.layers.LSTM(n_nodes, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer), return_sequences= True))
        model.add(tf.keras.layers.LSTM(n_nodes, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer), return_sequences= False))
        
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(1))

        model.compile(
            loss= tf.keras.losses.MeanSquaredError(),
            optimizer= tf.keras.optimizers.Adam(learning_rate= learning_rate),
            metrics= ["mse"]
        )

        model.fit(
            self.x_train,
            self.y_train,
            validation_data= (self.x_val, self.y_val),
            epochs= 100,
            batch_size= 32,
            callbacks= [tf.keras.callbacks.EarlyStopping(
                monitor= "val_loss",
                patience=10,
                restore_best_weights= True
            )]
        )

        acc_score= mean_squared_error(self.y_val, model.predict(self.x_val))

        return acc_score
    
    def objective(self, parent_run):
        with mlflow.start_run(run_id= parent_run.info.run_id, nested=True):
            study= optuna.create_study(direction= "minimize")
            study.optimize(self.objective_model, n_trials= 25, n_jobs= 4)

        return study
    

class ModelTrain:
    def __init__(self, train_path, val_path, seq):
        self.train_path= train_path
        self.val_path= val_path
        self.seq= seq
    
    def make_sequence(self, df):
        x, target= [], []
        
        for i in range(len(df)-self.seq):
            s= df.iloc[i: i+ self.seq]
            t= df.iloc[i+self.seq, 0]

            x.append(s)
            target.append(t)
        return np.array(x), np.array(target)
    
    def new_model(self, params):
        model= tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape= (7, 4)))

        model.add(tf.keras.layers.LSTM(params["n_nodes"], activation="relu",input_shape= (7, 4), kernel_regularizer=tf.keras.regularizers.l2(params["l2_regularizer"]), return_sequences= True))
        model.add(tf.keras.layers.LSTM(params["n_nodes"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["l2_regularizer"]), return_sequences= True))
        model.add(tf.keras.layers.LSTM(params["n_nodes"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["l2_regularizer"]), return_sequences= True))
        model.add(tf.keras.layers.LSTM(params["n_nodes"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["l2_regularizer"]), return_sequences= False))
        
        model.add(tf.keras.layers.Dropout(params["dropout"]))
        model.add(tf.keras.layers.Dense(32, activation= "relu"))
        model.add(tf.keras.layers.Dense(1))

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer= tf.keras.optimizers.Adam(learning_rate= params["learning_rate"]),
            metrics= ["mse"]
        )

        return model

    def metrics_score(self, ytrue, ypred, exp_type):
        return {
            f"{exp_type}_r2_score": r2_score(ytrue, ypred),
            f"{exp_type}_mse": mean_squared_error(ytrue, ypred)
        }

    def train(self):
        mlflow.set_tracking_uri= tracking_uri

        scaler= StandardScaler()
        data= pd.read_csv(self.train_path, index_col= "date", parse_dates= True)
        df_train= data.loc[: '2016-06']
        df_val= data.loc['2016-06':]

        df_train= pd.DataFrame(scaler.fit_transform(df_train), columns= scaler.get_feature_names_out())
        df_val= pd.DataFrame(scaler.transform(df_val), columns= scaler.get_feature_names_out())

        ds_train, target_train= self.make_sequence(df_train)
        ds_val, target_val= self.make_sequence(df_val)

        ds_train= ds_train.reshape(ds_train.shape[0], self.seq, 4)
        ds_val= ds_val.reshape(ds_val.shape[0], self.seq, 4)

        logger.info("Hyperparameter tuning started...")

        with mlflow.start_run() as parent_run:
            optuna_tuning= OptunaTuning(ds_train, target_train, ds_val, target_val)
            study= optuna_tuning.objective(parent_run)

        
        logger.info("Hyperparameter Tuning completed...")

        mlflow.log_params(study.best_params)
        best_model= self.new_model(study.best_params)

        history= best_model.fit(
            ds_train, 
            target_train,
            epochs= 100,
            batch_size= 32,
            validation_data= (ds_val, target_val),
            callbacks= [
                tf.keras.callbacks.EarlyStopping(
                    monitor= "val_loss",
                    patience= 10,
                    restore_best_weights= True
                )
            ]
        )

        ypred= best_model.predict(ds_val)
        val_metrics= self.metrics_score(target_val, ypred, "train")

        mlflow.log_metrics(val_metrics)

        plt.plot(history.history["mse"], color= "red", label="train")
        plt.plot(history.history["val_mse"], color="blue", label="validation")
        plt.legend()
        plt.grid(True)
        train_plot_path= "training_plot.jpg"
        plt.savefig(train_plot_path)
        plt.close()

        mlflow.log_artifact(train_plot_path)
        os.remove(train_plot_path)
        signature= infer_signature(ds_train, best_model.predict(ds_train))
        tf_model= mlflow.tensorflow.log_model(
                best_model,
                "tf_lstm_model",
                signature= signature,
                registered_model_name= "tf_lstm_model"
        )
        
        sklearn_model= mlflow.sklearn.log_model(
                scaler,
                "sklearn_scaler",
                
        )
        logger.info("Training completed...")


        # evaluation
        #run_id= mlflow.search_runs(filter_string= "run_name='train_run'")['run_id']
        model= mlflow.pyfunc.load_model(tf_model.model_uri)
        scaler= mlflow.sklearn.load_model(sklearn_model.model_uri)

        data= pd.read_csv(self.test_path, index_col= "date", parse_dates=["date"])
        #df_test= pd.DataFrame(scaler.transform(data), columns= scaler.get_feature_names_out())
        
        df_x, df_y= self.make_sequence(data)
        df_x= df_x.reshape(df_x.shape[0], self.seq, 4)
        y_pred= model.predict(df_x)
        
        #y_true= inverse_transform(df_y, scaler)
        #y_pred= inverse_transform(y_pred, scaler)

        test_metrics= self.metrics__score(df_y, y_pred, "test")
        mlflow.log_metrics(test_metrics)
        
        logger.info("Evaluation success...")


if __name__=="__main__":
    model_trainer= ModelTrain(TRAIN_FILE_PATH, TEST_FILE_PATH, 7)
    model_trainer.train()