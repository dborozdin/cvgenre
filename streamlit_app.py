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


@st.cache_data
def get_data():
    """Generate random sales data for Widget A through Widget Z"""

    product_names = ["Widget " + letter for letter in string.ascii_uppercase]
    average_daily_sales = np.random.normal(1_000, 300, len(product_names))
    products = dict(zip(product_names, average_daily_sales))

    data = pd.DataFrame({})
    sales_dates = np.arange(date(2023, 1, 1), date(2024, 1, 1), timedelta(days=1))
    for product, sales in products.items():
        data[product] = np.random.normal(sales, 300, len(sales_dates)).round(2)
    data.index = sales_dates
    data.index = data.index.date
    return data


@st.experimental_fragment
def show_daily_sales(data):
    time.sleep(1)
    with st.container(height=100):
        selected_date = st.date_input(
            "Pick a day ",
            value=date(2023, 1, 1),
            min_value=date(2023, 1, 1),
            max_value=date(2023, 12, 31),
            key="selected_date",
        )

    if "previous_date" not in st.session_state:
        st.session_state.previous_date = selected_date
    previous_date = st.session_state.previous_date
    st.session_state.previous_date = selected_date
    is_new_month = selected_date.replace(day=1) != previous_date.replace(day=1)
    if is_new_month:
        st.rerun()

    with st.container(height=510):
        st.header(f"Best sellers, {selected_date:%m/%d/%y}")
        top_ten = data.loc[selected_date].sort_values(ascending=False)[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(top_ten)
        cols[1].bar_chart(top_ten)

    with st.container(height=510):
        st.header(f"Worst sellers, {selected_date:%m/%d/%y}")
        bottom_ten = data.loc[selected_date].sort_values()[0:10]
        cols = st.columns([1, 4])
        cols[0].dataframe(bottom_ten)
        cols[1].bar_chart(bottom_ten)


def show_monthly_sales(data):
    time.sleep(1)
    selected_date = st.session_state.selected_date
    this_month = selected_date.replace(day=1)
    next_month = (selected_date.replace(day=28) + timedelta(days=4)).replace(day=1)

    st.container(height=100, border=False)
    with st.container(height=510):
        st.header(f"Daily sales for all products, {this_month:%B %Y}")
        monthly_sales = data[(data.index < next_month) & (data.index >= this_month)]
        st.write(monthly_sales)
    with st.container(height=510):
        st.header(f"Total sales for all products, {this_month:%B %Y}")
        st.bar_chart(monthly_sales.sum())


st.set_page_config(layout="wide")

st.title("Определение жанра музыки по фото обложки альбома")
st.markdown("На текущий момент может определять 10 жанров (аниме, блэк метал, классика, кантри, диско, ЭДМ, джазз, поп, рэп, рэгги)")

uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)

if uploaded_files:
   for uploaded_file in uploaded_files:
        st.write("Filename: ", uploaded_file.name)
        vector_2_use_filename='vector10.index'
        target_genres_csv_filename='train_genres10.csv'
        target_train= pd.read_csv(target_genres_csv_filename)
        
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
        
        test_file_path= uploaded_file.name
        st.write("Filename: ", test_file_path)
        
        image = Image.open(test_file_path)
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
        st.write('predicted genre:', top_genre)
