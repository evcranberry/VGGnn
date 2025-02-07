import streamlit as st

import requests
from io import BytesIO

from PIL import Image
import PIL
import torch
import torchvision
from torchvision import models, transforms
import io
import torch.nn as nn
import time


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
]
)

idx = {0: 'Вишня',
       1: 'Кофейное дерево',
       2: 'Огурец',
       3: 'Лисий орех (Макхана)',
       4: 'Лимон',
       5: 'Оливковое дерево',
       6: 'Жемчужное просо (Баджра)',
       7: 'Табак',
       8: 'Миндаль',
       9: 'Банан',
       10: 'Кардамон',
       11: 'Чили',
       12: 'Гвоздика',
       13: 'Кокос',
       14: 'Хлопок',
       15: 'Нут',
       16: 'Сорго',
       17: 'Джут',
       18: 'Кукуруза',
       19: 'Горчица',
       20: 'Папайя',
       21: 'Ананас',
       22: 'Рис',
       23: 'Соя',
       24: 'Сахарный тростник',
       25: 'Подсолнечник',
       26: 'Чай',
       27: 'Помидор',
       28: 'Вигна лучистая (Маш)',
       29: 'Пшеница'
       }

st.page_link('pages/Агро_метрики_модели.py', label='Узнать детали обучения модели')

st.markdown('''
    ## :green[🌾Сельскохозяйственные культуры]
    ##### :green[🖼️Определяем культуру по фото]
    ''')



st.logo('./images/icons/sunflower.jpg', icon_image='./images/icons/plant.jpg', size='large')

st.markdown('''
#### <- Загрузите фото с растением в боковую панель
''')


uploaded_files = st.sidebar.file_uploader(
    "Выберите фото", type=['png', 'jpg'], accept_multiple_files=True
)
url_file = st.sidebar.text_input("Вставьте ссылку на картинку",)
url_img = None
if url_file:
    try:
        url_img = Image.open(BytesIO(requests.get(url_file).content)).convert("RGB")
    except PIL.UnidentifiedImageError:
        st.sidebar.write("❌Некорректная ссылка")
model = torchvision.models.resnet152()
model.fc = nn.Linear(in_features=2048, out_features=30, bias=True)
model.load_state_dict(torch.load('models/resnet152_agriculture.pt'))
model.eval()
if uploaded_files:
    image_tensors = []
    images = []
    if url_img:
        image_tensors.append(preprocess(url_img))
        images.append(url_img)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        # Преобразование изображения
        image_tensor = preprocess(image)
        # print(image_tensor.shape)
        images.append(image)
        image_tensors.append(image_tensor)

    # Объединение всех тензоров в один батч
    input_batch = torch.stack(image_tensors)
    # print(input_batch.shape)
    # Передача батча в модель

    start_time = time.time()
    with torch.no_grad():
        output = model(input_batch)
    end_time = time.time()

    elapsed_time = end_time - start_time
    st.write(f"⏳Время выполнения: {elapsed_time:.4f} секунд")
    # Получение предсказаний
    _, predicted_idxs = torch.max(output, 1)
    for i, uploaded_file in enumerate(images):
        st.image(images[i], use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 24px; font-family: sans-serif;">
                <p>Это {idx[predicted_idxs[i].item()]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


if 'show_text' not in st.session_state:
    st.session_state.show_text = False

if st.sidebar.button("Наша команда"):
    st.session_state.show_text = not st.session_state.show_text

if st.session_state.show_text:
    st.sidebar.markdown('''
    * 🌺 Masha
    * 🌼 Nanzat
    ''')


st.sidebar.link_button("Наш Github", "https://github.com/evcranberry/VGGnn")
