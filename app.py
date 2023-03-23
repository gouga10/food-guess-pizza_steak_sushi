import gradio as gr 
import os
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple,Dict

class_names=['pizza','steak','sushi']
effnetb2,effnetb2_transforms=create_effnetb2_model(
    num_classes=3
)

effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent",
        map_location=torch.device('cpu')
    )
)

def predict(img) ->Tuple[Dict,float]:
  start_timer=timer()
  img=effnetb2_transforms(img).unsqueeze(0)
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs=torch.softmax(effnetb2(img),dim=1)
  
  pred_labels_and_probs={class_names[i]:float(pred_probs[0][i])for i in range(len(class_names))}
  pred_time=round(timer()-start_timer,5)
  return pred_labels_and_probs,pred_time

title='food_vis_minii :)'
description='an efffnetb2 feature extractor'
article='created in lesson 9 ML course'

example_list=[["examples/"+example]for example in os.listdir('examples')]

demo=gr.Interface(fn=predict,
                  inputs=gr.Image(type='pil'),
                  outputs=[gr.Label(num_top_classes=3,label='predictions'),
                           gr.Number(label='PRediction time(s)')],
                  examples=example_list,
                  title=title,
                  description=description,
                  article=article)



demo.launch()








