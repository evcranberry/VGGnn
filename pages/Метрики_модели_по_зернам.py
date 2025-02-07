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

st.title('Модель VGG, обученная классификации кофейных зёрен')

st.write('### Процесс обучения модели:')

st.image('fit.png')

st.write('#### Время обучения модели на 15 эпохах:')

st.write('''
#### Состав датасета: 
* __4 класса зёрен:__ тёмные, светлые, зелёные и средние  
* __тренировочный датасет:__ 1200 элементов (по 300 каждого класса)
* __тестовый датасет:__ 400 элементов (по 100 каждого класса)''')