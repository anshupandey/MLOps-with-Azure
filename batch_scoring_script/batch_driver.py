import os
import pandas as pd
import joblib
import logging
import numpy as np


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),"outputs/model.pkl") # AZUREML_MODEL_DIR = ./azureml-models/MODEL_NAME/VERSION
    model = joblib.load(path)
    logging.info("initialization completed")

def run(mini_batch) -> pd.DataFrame:
    print(f"Executing run method over batch of {len(mini_batch)} files.")

    results = []
    for data_batch in mini_batch:
        logging.info(data_batch)
        # Read comma-delimited data into an array
        data = np.genfromtxt(data_batch, delimiter=',')
        logging.info(data)
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append({"predictions":prediction})

    return pd.DataFrame(results)
