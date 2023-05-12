
import os
import joblib
import json, logging, numpy
def init():
    global model
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),"./model.pkl") # AZUREML_MODEL_DIR = ./azureml-models/MODEL_NAME/VERSION
    model = joblib.load(path)
    loggign.info("initialization completed")

def run(raw_data):
    logging.info("model: request received")
    data = json.loads(raw_data)['data']
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("request procsesed")
    return json.dumps(result.tolist())

