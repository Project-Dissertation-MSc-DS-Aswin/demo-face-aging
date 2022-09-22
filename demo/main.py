"""
Adapted from Source: 
https://testdriven.io/blog/fastapi-streamlit/
"""

# Import Necessary Libraries

import uuid
import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI, Header
from fastapi import UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import pandas as pd
import config
import inference
import drift
import app_preprocessing
import context
import os
from time import time
import pickle
from config import *
from time import time
from typing import Union, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import socketio
import config
import subprocess
import pathlib
from pathlib import Path, PurePath

app = FastAPI()

origins = [
    "https://bejewelled-pie-d79a27.netlify.app", 
    "http://bejewelled-pie-d79a27.netlify.app", 
    "http://20.120.53.95:8080",
    "https://20.120.53.95:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("../src/data_collection/agedb_drift_beta_optimized.csv")

def map_pipeline(pipeline):
  if pipeline == "face_aging":
    return "Aging"
  return ""

"""
REST API:
GET /
    - Displays the message Welcome from the API
"""
@app.get("/")
def read_root():
  return {"message": "Welcome from the API"}


"""
REST API:
POST /model/classification
    - Uploaded File
POST /model/anomaly_detection
    - Uploaded File
"""
@app.get("/images/temp/{images_path}")
async def get_image(images_path: str):
  def iterfile():
    with open('images/temp/' + images_path, mode="rb") as file_like:
      yield from file_like

  return StreamingResponse(iterfile(), media_type="image/jpg")

@app.get("/backend/images/{num}/")
def get_image_url(num: int):
  paths = []
  np.random.seed(1000)
  drifted_df = df.loc[df['drifted'] == 0].sample(num//2)
  np.random.seed(1000)
  non_drifted_df = df.loc[df['drifted'] == 1].sample(num//2)
  dataframe = pd.concat([drifted_df, non_drifted_df], axis=0)
  image_paths = dataframe.filename
  drifted = dataframe.drifted.values.tolist()
  
  for image_path in image_paths.values:
    image_path = image_path.replace(".jpg", "-resized.jpg")
    paths.append("""images/temp/{image}""".format(image=image_path))
  
  return {"paths": paths, "drifted": drifted}

class Item(BaseModel):
  filenames: List[str] = []

# ml_model_classification = pickle.load(open("../src/models/agedb_voting_classifier_age.pkl", 'rb'))

@app.post("/backend/drift/{seed}/{num}/{pipeline}/")
def get_drift_predictions(seed: int, num: int, pipeline: str, item: Item):
  
  DIRECTORY = os.curdir
  parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(DIRECTORY)))
  path = PurePath(parent_dir)
  pipeline_dir = path.parts
  idx = list(pipeline_dir).index("demo")
  pipeline_dir = os.sep.join(pipeline_dir[0:idx]) + os.sep + os.sep.join(pipeline_dir[idx+1:])
  
  python_exe = os.environ["PYTHON_EXE"]
  
  filenames = item.filenames
  
  drift_source_filenames = pd.read_csv("../src/data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference.csv")
  idx = [drift_source_filenames['filename'] == filename for filename in filenames]
  result_idx = [False]*len(drift_source_filenames)
  for i in idx:
      result_idx = np.logical_or(result_idx, i)
  
  drift_source_filenames = drift_source_filenames.loc[result_idx].reset_index()
  drift_source_filenames.to_csv("../src/data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference_reduced.csv")
  
  script_path = ["face_with_aging.py", "dataset=agedb", "model_path=../models/facenet_keras.h5", "batch_size=128", "metadata=../dataset_meta/AgeDB_metadata.mat", 
  "no_of_samples=20", "no_of_pca_samples=20", "grouping_distance_type=DISTINCT", "tracking_uri=mlruns/", "logger_name=facenet_with_aging", 
  "classifier=../models/16.07.2022_two_classifiers_facenet/facenet_agedb_voting_classifier_age_test_younger_latest.pkl", "experiment_id=0", "pca_type=KernelPCA", "noise_error=0", 
  "mode=image_reconstruction", "drift_beta=1.0", "covariates_beta=0", "data_dir=../../demo/images/temp", 
  "drift_synthesis_filename=../data_collection/facenet_agedb_drift_synthesis_filename-range-of-beta-latest-1.csv", 
  "drift_source_filename=../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference.csv", 
  "drift_source_reduced_filename=../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference_reduced.csv", 
  "model=FaceNetKeras", "input_shape=-1,160,160,3", "function_type=morph", 
  "drift_type=incremental"]
  
  print("""cd ..\\src\\pipeline && {python_exe} {script_path}""".format(python_exe=python_exe, script_path=" ".join(script_path)))
  subprocess.run([python_exe] + script_path, shell=True, cwd='../src/pipeline')
  
  args = context.Args({})
  args.metadata = "../src/dataset_meta/AgeDB_metadata.mat"
  args.classifier = "../src/models/16.07.2022_two_classifiers_facenet/facenet_agedb_voting_classifier_age_test_younger_latest.pkl"
  args.batch_size = 128
  args.dataset = "agedb"
  args.input_shape = (-1,160,160,3)
  args.alt_input_shape = (-1,96,96,3)
  args.model = "FaceNetKeras"
  args.data_dir = "../demo/images/temp"
  args.pca_type = 'KernelPCA'
  args.grouping_distance_type = 'DISTINCT'
  args.mode = 'image_reconstruction'
  args.drift_type = 'incremental'
  args.model_path = "../src/models/facenet_keras.h5"
  args.batch_size = 128
  args.input_shape = (-1,160,160,3)
  args.function_type = 'morph'
  args.alt_input_shape = (-1,96,96,3)
  args.drift_beta = 1
  args.denoise_type = 'opencv_denoising'
  args.drift_source_filename = "../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference_reduced.csv"
  
  preprocessor = app_preprocessing.Preprocessor()
  
  images, labels = preprocessor.prepare_data(filenames)
  
  _, reconstructed_images, original_images = preprocessor.reconstruct_image(args, images, None, filenames)
  
  dataframe = pd.read_csv(os.path.join("../src/data_collection", "facenet_agedb_drift_synthesis_filename-range-of-beta-latest-1.csv"))
  
  pd.options.mode.use_inf_as_na = True
  
  drift_properties = drift.DriftProperties()
  drift_features = drift_properties.extract_statistical_properties(args, preprocessor.experiment, images, reconstructed_images, original_images)
  
  df = pd.concat([dataframe, pd.DataFrame(np.concatenate(drift_features).reshape(-1,4), columns=['mse_p', 'mse_t', 'mse_corr', 'psnr'])], axis=1).reset_index()
  
  c = df.shape[0]//11
  for i in range(df.shape[0]//11):
    df.loc[11*i:11*i+11, 'mse_p'] = (df.loc[11*i:11*i+11, 'mse_p'] - df.loc[11*i:11*i+11, 'mse_p'].mean(skipna=True)) / df.loc[11*i:11*i+11, 'mse_p'].std(skipna=True)
    df.loc[11*i:11*i+11, 'psnr'] = (df.loc[11*i:11*i+11, 'psnr'] - df.loc[11*i:11*i+11, 'psnr'].mean(skipna=True)) / df.loc[11*i:11*i+11, 'psnr'].std(skipna=True)
  
  df, reconstructed_images, original_images = preprocessor.reduce(args, reconstructed_images, original_images, df)
  
  print(df.columns)
  
  data = df.loc[:, ['mse_p', 'psnr', 'drift_beta', 'age', 'drift_difference', 'filename', 'drifted']]
  
  train_data, train_labels = data[['mse_p', 'psnr', 'drift_beta', 'age', 'drift_difference']], data['drifted']
  
  print(train_data)
  
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  
  p = make_pipeline(
      StandardScaler(),
      PCA(n_components=2)
  )
  
  X_pca = p.fit_transform(train_data)
  
  fig, ax = plt.subplots(1,1,figsize=(12,8))
  ax.scatter(X_pca[train_labels.values==0, 0], X_pca[train_labels.values==0, 1], color='blue', label='Non-Drifted Images')
  ax.scatter(X_pca[train_labels.values==1, 0], X_pca[train_labels.values==1, 1], color='orange', label='Drifted Images')
  ax.legend()
  fig.savefig("plot.png")
  
  return {"predictions": data.to_dict(), "score_test": 0, "score_validation": 0}
  
if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8080)