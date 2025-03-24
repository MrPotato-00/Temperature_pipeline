import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.paths_config import *

import pickle

with open(PyTorch_Params, "rb") as f:
    params= pickle.load(f)

print(params)