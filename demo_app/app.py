import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBRegressor


# File Paths
model_path = 'xgbr_model.sav'
component_config_path = "component_configs.json"
examples_path = "examples.pkl"

# predefined
feature_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']

# Loading the files
model = pickle.load(open(model_path, 'rb'))
with open(examples_path,"rb") as f: examples = pickle.load(f) 
feature_limitations = json.load(open(component_config_path, "r"))


# Util function
def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = np.array(features).reshape(-1,len(feature_order))

  # prediction
  pred = model.predict(features) # .predict_proba(features)

  return np.round(pred,3)

# Creating the gui component according to component.json file
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )



# creating the app
demo_app = gr.Interface(predict, inputs, "number",examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()