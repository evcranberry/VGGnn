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


model1 = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.DEFAULT)
model2 = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.DEFAULT)
model1.eval()
model2.eval()

with open('./models/imagenet1000_clsidx_to_labels.txt', "r") as file:
    json_str = file.read().splitlines()

labels_1000 = list(map(lambda x: x.split("\'")[1].split(",")[0], json_str))
idx = {i: k for i, k in enumerate(labels_1000)}
# print(idx)
preprocess1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ]
)
preprocess2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ]
)


st.markdown('''
    ## :blue[🖼️Классификатор изображений]
    ''')



st.logo('./images/icons/rob.png', icon_image='./images/icons/robot.jpg', size='large')

st.markdown('''
#### <- Загрузите фото в боковую панель
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


if uploaded_files:
    image_tensors = []
    images = []
    if url_img:
        image_tensors.append(preprocess1(url_img))
        images.append(url_img)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        # Преобразование изображения
        image_tensor = preprocess1(image)
        # print(image_tensor.shape)
        images.append(image)
        image_tensors.append(image_tensor)

    # Объединение всех тензоров в один батч
    input_batch = torch.stack(image_tensors)
    # print(input_batch.shape)
    # Передача батча в модель

    start_time = time.time()
    with torch.no_grad():
        output1 = torch.nn.functional.softmax(model1(input_batch))
        output2 = torch.nn.functional.softmax(model2(input_batch))
        # print(output1.shape)
    end_time = time.time()

    elapsed_time = end_time - start_time
    st.write(f"⏳Время выполнения: {elapsed_time:.4f} секунд")

    # Получение топ-5 предсказаний для каждой модели
    top5_probs1, top5_idxs1 = torch.topk(output1, 5, dim=1)
    top5_probs2, top5_idxs2 = torch.topk(output2, 5, dim=1)

    for i, uploaded_file in enumerate(images):
        st.image(images[i], use_container_width=True)

        # Топ-5 классов для первой модели
        top5_classes1 = [idx[int(id)] for id in top5_idxs1[i]]
        top5_probs_list1 = top5_probs1[i].tolist()

        # Топ-5 классов для второй модели
        top5_classes2 = [idx[int(id)] for id in top5_idxs2[i]]
        top5_probs_list2 = top5_probs2[i].tolist()

        # Объединение результатов в пары
        top5_pairs = [
            (top5_classes1[j], top5_probs_list1[j], top5_classes2[j], top5_probs_list2[j])
            for j in range(5)
        ]

        st.markdown(
            f"""
            <div style="text-align: center; font-size: 24px; font-family: sans-serif;">
                <p>Top5 classes (ViT B 16 | EfficientNet v2 L):</p>
                <ul>
                    <li>{top5_pairs[0][0]}: {top5_pairs[0][1]:.4f} | {top5_pairs[0][2]}: {top5_pairs[0][3]:.4f}</li>
                    <li>{top5_pairs[1][0]}: {top5_pairs[1][1]:.4f} | {top5_pairs[1][2]}: {top5_pairs[1][3]:.4f}</li>
                    <li>{top5_pairs[2][0]}: {top5_pairs[2][1]:.4f} | {top5_pairs[2][2]}: {top5_pairs[2][3]:.4f}</li>
                    <li>{top5_pairs[3][0]}: {top5_pairs[3][1]:.4f} | {top5_pairs[3][2]}: {top5_pairs[3][3]:.4f}</li>
                    <li>{top5_pairs[4][0]}: {top5_pairs[4][1]:.4f} | {top5_pairs[4][2]}: {top5_pairs[4][3]:.4f}</li>
                </ul>
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
