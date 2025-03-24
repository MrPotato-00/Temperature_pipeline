import os

RAW_DIR= "artifacts/raw"
TRAIN_FILE_PATH= os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH= os.path.join(RAW_DIR, "test.csv")

PROCESSED_DIR= "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH= os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH= os.path.join(PROCESSED_DIR, "processed_test.csv")

MODEL_DIR= "artifacts/model"
PyTorch_Params= os.path.join(MODEL_DIR, "best_params.pkl")
PyTorch_MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "pytorch_lstm_model.pth")
SKLEARN_MODEL_OUTPUT_PATH= os.path.join(MODEL_DIR, "sklearn_model.pkl")


