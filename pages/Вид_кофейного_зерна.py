import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import time
import urllib.request


st.page_link('pages/Кофе_метрики_модели.py', label='Узнать детали обучения модели')
st.title('__Определите вид Вашего кофейного зерна!☕️__')

st.logo('./images/icons/cup.jpg', icon_image='./images/icons/bean.jpeg', size='large')

from torchvision.models import vgg19_bn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VGG19_BN_Weights_custom = torch.load('models/weights_vgg.pt', map_location=torch.device(DEVICE))
model = vgg19_bn(weights=VGG19_BN_Weights_custom)
model.to(DEVICE)

model.classifier[6] = nn.Linear(4096, 4)
model.eval()
coffee_types = {0: 'Тёмный', 1: 'Зеленый', 2: 'Светлый', 3: 'Средний'}
trnsfrms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor()
    ]
)

def get_prediction(img) -> int:
    img = trnsfrms(img)  # Применяем трансформации
    img_display = torch.permute(img, (1, 2, 0)).numpy()
    start = time.time()
    with torch.inference_mode():
        output = model(img.unsqueeze(0).to(DEVICE))  # Получаем вывод модели
        end = time.time()
        pred_class = torch.argmax(output, dim=1).item()  # Получаем индекс класса с наибольшей вероятностью
    return coffee_types[pred_class], img_display, end-start

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.write('##### <- Сделайте фото зерна, как на примере ниже, и загрузите его в боковую панель:')
ex_image = Image.open('images/light (5).png')
st.image(ex_image)

uploaded_file = st.sidebar.file_uploader(label='Загружать фото сюда:', type=['jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    pred, img, sec = get_prediction(image)
    st.session_state.predictions.append((pred.lower(), img))
    for pred, img in st.session_state.predictions:
        st.write(f'''Тип Вашего кофейного зерна: _{pred}_  
Время выполнения предсказания: __{sec:.4f} секунды__''')
        st.image(img)

link = st.sidebar.text_input(label='Вставьте сюда ссылку на картинку зерна')
if link is not '':
    image = Image.open(urllib.request.urlopen(link))
    pred, img, sec = get_prediction(image)
    st.session_state.predictions.append((pred.lower(), img))
    for pred, img in st.session_state.predictions:
        st.write(f'''Тип Вашего кофейного зерна: _{pred}_  
Время выполнения предсказания: __{sec:.4f} секунды__''')
        st.image(img)

