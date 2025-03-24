import os


MODEL_DIR= "model"
PyTorch_Params= os.path.join(MODEL_DIR, "best_params.pkl")
PyTorch_MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "pytorch_lstm_model.pth")
SKLEARN_MODEL_OUTPUT_PATH= os.path.join(MODEL_DIR, "sklearn_model.pkl")


