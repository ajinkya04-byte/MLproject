import os
import sys
from src.logger import logging
from src.exception import custom_execption

import dill
import numpy as np 
import pandas as pd
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            dill.dump(obj,file)
    except Exception as e:
        raise custom_execption(e,sys)