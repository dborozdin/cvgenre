import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import time
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import os
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
import fastai
from fastai.vision.all import *
import pathlib

st.set_page_config(layout="wide")

st.title("Определение жанра музыки по фото обложки альбома")
st.markdown("На текущий момент может определять 10 жанров (аниме, блэк метал, классика, кантри, диско, ЭДМ, джаз, поп, рэп, рэгги)")

uploaded_files = st.file_uploader("Загрузите файлы...", accept_multiple_files=True)
useDemoPictures = st.checkbox('Использовать демонстрационные данные')
canUseFASTAI=st.checkbox('Использовать модель FASTAI')
demo_files_list=['test_img1.png', 'test_img2.png', 'test_img3.png']

if uploaded_files or useDemoPictures:
   vector_2_use_filename='vector10.index'
   target_genres_csv_filename='train_genres10.csv'
   target_train= pd.read_csv(target_genres_csv_filename)
   device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
   processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
   model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

   if canUseFASTAI:
       EXPORT_PATH = pathlib.Path("./fastai_model10.pkl")
       learn_inf=None
       try:
            learn_inf = load_learner(EXPORT_PATH)
       except ValueError:
            st.write('Ошибка загрузки модели:', ValueError)
            canUseFASTAI=False
   if useDemoPictures:
        for demo_file in demo_files_list:
            uploaded_files.append(open(demo_file, 'rb'))
        
        
   for uploaded_file in uploaded_files:
        test_file_path= uploaded_file.name
        image_data=None
        if useDemoPictures:
            image_data = uploaded_file.read()
        else:
            image_data = uploaded_file.getvalue()
        st.image(image_data)
        image= Image.open(BytesIO(image_data))
        #Extract the features
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
        #Normalize the features before search
        embeddings = outputs.last_hidden_state
        embeddings = embeddings.mean(dim=1)
        vector = embeddings.detach().cpu().numpy()
        vector = np.float32(vector)
        faiss.normalize_L2(vector)
        #Read the index file and perform search of top-1 images
        index = faiss.read_index(vector_2_use_filename)
        d,i = index.search(vector,5)
        similar_elements_genres_arr=[]
        #находим запись по индексу в тренировочном датафрейме и берем жанр наиболее часто встречающийся
        for res in i[0]:
            element= target_train.iloc[[res]]
            genre= element['genre'].tolist()[0]
            similar_elements_genres_arr.append(genre)
        top_genre = Counter(similar_elements_genres_arr).most_common(1)[0][0]
        st.write('FAISS: эта обложка относится к музыке жанра:', top_genre)
       
        #fastAI
        if canUseFASTAI:
            predictions=learn_inf.predict(image.convert('RGB'))
            st.write('FAISS: эта обложка относится к музыке жанра:', predictions[0])

      
