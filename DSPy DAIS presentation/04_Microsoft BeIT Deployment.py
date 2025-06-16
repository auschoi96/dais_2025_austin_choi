# Databricks notebook source
# MAGIC %md
# MAGIC #Introduction
# MAGIC
# MAGIC This notebook demonstrates how you can load a model from huggingface. We will use the vision transformer BEiT from Microsoft to do image classification. You can view the model card here: https://huggingface.co/microsoft/beit-base-patch16-224
# MAGIC
# MAGIC ### Prerequisites
# MAGIC Ensure you are on **Databricks Runtime 16.3 ML LTS**. Use at least m5d2xlarge compute to load the model

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow transformers==4.42.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Testing the model 
# MAGIC
# MAGIC This cell simply loads the model and tests to see how the model works

# COMMAND ----------

from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import requests

url = 'https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/EK6252.JPG'
image = Image.open(requests.get(url, stream=True).raw)

processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


# COMMAND ----------

# MAGIC %md
# MAGIC #MLFlow and Unity Catalog Registration
# MAGIC
# MAGIC We use mlflow, the transformers pipeline class and unity catalog to load and register the model. The registered_model_name tells mlflow where to save and register the model. If you rerun the cell, you will see a new version will be created instead of replacing the model. This is helpful when you do training runs and would like to save different versions of your models for scoring
# MAGIC
# MAGIC When you log models that require more dependencies or private dependencies, there are options like conda_env or pip_requirements that you can use to specify said dependencies. Databricks does this by default when using Databricks Runtime ML 

# COMMAND ----------

from config import volume_label, volume_name, catalog, schema, model_name, model_endpoint_name, embedding_table_name, embedding_table_name_index, registered_model_name, vector_search_endpoint_name, beit_model_name

# COMMAND ----------

import mlflow
import transformers
from transformers import pipeline

pipe = pipeline("image-classification", model="microsoft/beit-base-patch16-224")
catalog = catalog 
schema = schema 
model = beit_model_name #change this to the name you would like to call the model

with mlflow.start_run():
    model_info = mlflow.transformers.log_model( #there are other flavors of MLflow that you can use to help with this process
        transformers_model=pipe,
        artifact_path="vision-model",
        task="vision",
        registered_model_name=f"{catalog}.{schema}.{model}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #Test the model works
# MAGIC
# MAGIC We can load the model and test it locally (or on this current notebook's compute) to see if it works. If it does work, it is ready to be served. 
# MAGIC
# MAGIC You can also use this function to load the model and use it in your workloads without serving it but, bear in mind, this will load it in memory and you will have to reload it each time.

# COMMAND ----------

loaded_model = mlflow.transformers.load_model(model_info.model_uri)
url = 'https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/JO9469.JPG'
result = loaded_model(url)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC #Deploy the Model (Optional)
# MAGIC
# MAGIC You can utilize model serving to load your model onto CPU or GPU resources so that it is immediately accessible. Model Serving will handle the environment and endpoint for you, even providing a REST API for you to use to call the model
# MAGIC
# MAGIC You can deploy the model through the UI (on the left under serving) or programatically below 

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name=model,
    config={
        "served_entities": [
            {
                "name": model,
                "entity_name": f"{catalog}.{schema}.{model}",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": model,
                    "traffic_percentage": 100
                }
            ]
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Hit your endpoint
# MAGIC
# MAGIC Use the code below to programatically hit your newly created endpoint!

# COMMAND ----------

import json
import requests
# send the POST request to create the serving endpoint
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
DATABRICKS_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

"""The data must be in the JSON formats below to run inference"""

"""Batch inference with a column for tabularize the data"""

# data = {
#   "dataframe_split": { 
#     "columns": ["URL"],
#     "data": [['https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/JO9469.JPG','https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/DQ3639.JPG','https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241024truck/EK3333.JPG']]
#   }
# }
"""Batch inference with a list of inputs"""
# data = {
#   "inputs" : ['https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/JO9469.JPG','https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241016ve/DQ3639.JPG','https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241024truck/EK3333.JPG'],
#   "params" : {"max_new_tokens": 100, "temperature": 1}
# }
"""Quick inference on one input"""
data = {"inputs": ['https://d323w7klwy72q3.cloudfront.net/i/a/2024/20241022govt/LC9186.JPG']}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"https://{DATABRICKS_URL}/serving-endpoints/{model}/invocations", json=data, headers=headers
)

result2 = response.json()


# COMMAND ----------

result2['predictions'][0]['0']['label'] 

# COMMAND ----------


