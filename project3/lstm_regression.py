import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizer'ı kaydetmek ve yüklemek için

from sklearn.model_selection import train_test_split #veriyi eğitim ve test setlerine bölmek için
from sklearn.preprocessing import MinMaxScaler #veriyi ölçeklendirmek için

from tensorflow.keras.preprocessing.text import Tokenizer #metni sayısal dizilere dönüştürmek için
from tensorflow.keras.preprocessing.sequence import pad_sequences #dizileri aynı uzunluğa getirmek için, padding işlemi için
from tensorflow.keras.models import Sequential #model oluşturmak için
from tensorflow.keras.layers import Embedding, LSTM, Dense #model katmanları
from tensorlow.keras.losses import MeanSquaredError #regresyon için kayıp fonksiyonu    
from tensorlow.keras.metrics import MeanAbsoluteError #regresyon için metrik
