#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:39:21 2019

@author: aidanrobertson
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from flask import Flask, render_template, request
from fastai.basic_train import load_learner
from fastai.vision import *
import pandas as pd
import numpy as np
import torch
#from fastai.vision import open_image
class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
        
def get_output(module, input_value, output):
    return output.flatten(1)

def get_input(module, input_value, output):
    return list(input_value)[0]

def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None
# Create the application object
app = Flask(__name__)
learn = load_learner(path = '.',file = 'recommendermodel')
#img_repr_df = pd.read_csv('feature_vectors.csv')

#img_repr_df['img_repr'] = img_repr_df['img_repr'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
@app.route('/', methods=['GET', 'POST'])
#when reading the file, run this lambda function on the feature vector column to convert list of numbers with '\n' text to a 2d array

def classification():
  # Pull input
  if request.method == 'GET':
    return render_template('index.html')
  
  if request.method == 'POST':
    file = request.files['file']
    img_repr_df = pd.read_csv('feature_vectorslastone.csv')
    img_repr_df['img_repr'] = img_repr_df['img_repr'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
    x = open_image(file)
    model = learn.model
    linear_output_layer = get_named_module_from_model(model, '1.4')
    xb, _ = learn.data.one_item(x)
    #xb_im = Image(learn.denorm(xb)[0])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    xb = xb.to(device)
    with Hook(linear_output_layer, get_output, True, True) as hook:
        bs = xb.shape[0]
        result = model.eval()(xb)
        img_reprs = hook.stored.cpu().numpy()
        img_reprs = img_reprs.reshape(bs, -1)
    base_vector = img_reprs
    from scipy.spatial.distance import cosine
    cosine_similarity = 1 - img_repr_df['img_repr'].apply(lambda x: cosine(x, base_vector))
    similar_img_ids = np.argsort(cosine_similarity)[-5:][::-1]
    img_repr_df_new = img_repr_df.iloc[similar_img_ids].reset_index()
    
    product_id= img_repr_df_new['product_id']
    product_name = img_repr_df_new['product_name']
    price = img_repr_df_new['price_CAD']
    base_label= img_repr_df_new['collection']
    material = img_repr_df_new['material']
    main_img = img_repr_df_new['main_img']
    alt_img = img_repr_df_new['alt_img']
    product_link = img_repr_df_new['product_link']
    #export new model, model.model then rest of input image manipulation
    product_id1 = product_id[0]
    product_id2 = product_id[1]
    product_id3 = product_id[2]
    product_id4 = product_id[3]
    product_id5 = product_id[4]
    product_name1 = product_name[0]
    product_name2 = product_name[1]
    product_name3 = product_name[2]
    product_name4 = product_name[3]
    product_name5 = product_name[4]
    price1 = price[0]
    price2 = price[1]
    price3 = price[2]
    price4 = price[3]
    price5 = price[4]
    base_label1 =  base_label[0]
    base_label2 = base_label[1]
    base_label3 = base_label[2]
    base_label4 = base_label[3]
    base_label5 = base_label[4]
    material1 = material[0]
    material2 = material[1]
    material3 = material[2]
    material4 = material[3]
    material5 = material[4]
    mainimg1 = main_img[0]
    mainimg2 = main_img[1]
    mainimg3 = main_img[2]
    mainimg4 = main_img[3]
    mainimg5 = main_img[4]
    altimg1 = alt_img[0]
    altimg2 = alt_img[1]
    altimg3 = alt_img[2]
    altimg4 = alt_img[3]
    altimg5 = alt_img[4]
    productlink1 = product_link[0]
    productlink2 = product_link[1]
    productlink3 = product_link[2]
    productlink4 = product_link[3]
    productlink5 = product_link[4]
    return render_template("results.html",
                           Img1ID=product_id1, Img1Name=product_name1, Img1Price=price1,
                           Img1Label=base_label1, Img1Mat=material1, Img1main=mainimg1,Img1alt = altimg1,Img1link = productlink1,
                           Img2ID=product_id2, Img2Name=product_name2, Img2Price=price2,
                           Img2Label=base_label2, Img2Mat=material2, Img2main=mainimg2,Img2alt = altimg2,Img2link = productlink2,
                           Img3ID=product_id3, Img3Name=product_name3, Img3Price=price3,
                           Img3Label=base_label3, Img3Mat=material3, Img3main=mainimg3,Img3alt = altimg3,Img3link = productlink3,
                           Img4ID=product_id4, Img4Name=product_name4, Img4Price=price4,
                           Img4Label=base_label4, Img4Mat=material4, Img4main=mainimg4,Img4alt = altimg4,Img4link = productlink4,
                           Img5ID=product_id5, Img5Name=product_name5, Img5Price=price5,
                           Img5Label=base_label5, Img5Mat=material5, Img5main=mainimg5,Img5alt = altimg5,Img5link = productlink5
                           )


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=80) #will run locally http://127.0.0.1:5000/