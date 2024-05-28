# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import torch

import boto3
import numpy as np
from PIL import Image

import flask
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class SamService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with open(os.path.join(model_path, "vit_b_encoder_jit_py20.pt"), "rb") as f:
                cls.model = torch.jit.load(f, map_location=device)
        return cls.model

    @classmethod
    def predict(cls, input_tensor):
        clf = cls.get_model()

        # Inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device selected: {}".format(device))
        clf.to(device)
        clf.eval()
        torch_input_tensor = torch.tensor(input_tensor).to(device)
        with torch.no_grad():
            predictions = clf(torch_input_tensor)
        return predictions


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = SamService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    data = None

    if flask.request.content_type != 'application/json':
        raise Exception("File type not matching: {}".format(flask.request.content_type))
        return 

    s3client = boto3.client('s3')

    input_data = json.loads(flask.request.data)

    file_key = input_data["instances"][0]["data"]["key"]

    for instance in input_data["instances"]:
        bucket = instance["data"]["bucket"]
        key = instance["data"]["key"]

        image_data = s3client.get_object(Bucket=bucket, Key=key)["Body"].read()

        # Data Preprocessing
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))

        orig_width, orig_height = image.size
        resized_width, resized_height = image.size
        if orig_width > orig_height:
            resized_width = 1024
            resized_height = int(1024 / orig_width * orig_height)
        else:
            resized_height = 1024
            resized_width = int(1024 / orig_height * orig_width)
        image = image.resize((resized_width, resized_height), Image.BILINEAR)

        input_tensor = np.array(image)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([[58.395, 57.12, 57.375]])

        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.transpose(2,0,1)[None,:,:,:].astype(np.float32)

        if resized_height < resized_width:
            input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,1024-resized_height),(0,0)))
        else:
            input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,0),(0,1024-resized_width)))

        # Inference
        predictions = SamService.predict(input_tensor)
        
        # Output
        embeddings_npy = predictions.detach().cpu().numpy()
        np.save("embeddings.npy", embeddings_npy)

        key_parts = key.split("/")
        file_parts = key_parts.pop().split(".")
        file_parts[-1] = "npy"
        key_parts.append("sam_v1_encoding/" + ".".join(file_parts))
        new_key = "/".join(key_parts)

        try:
            response = s3client.upload_file("embeddings.npy", Bucket=bucket, Key=new_key)
            print("npy file saved to s3")
        except ClientError as e:
            print(e)
        os.remove("embeddings.npy")

    file_key_parts = file_key.split("/")
    file_key_parts.pop()
    file_key_parts.append("sam_v1_encoding")
    embeddings_directory_key = ("/").join(file_key_parts)
    res = json.dumps({"embedding_location": { "bucket": bucket, "key": embeddings_directory_key }})

    return flask.Response(response=res, status=200, mimetype="application/json")
