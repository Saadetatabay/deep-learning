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
from tensorflow.keras.losses import MeanSquaredError #regresyon için kayıp fonksiyonu    
from tensorflow.keras.metrics import MeanAbsoluteError #regresyon için metrik


splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
print(df.head())

# lablları 0-4 aralığından 1-5 aralığına çekiyoruz
df['label'] = df['label'] + 1

texts = df['text'].values
print(texts[:5])
labels = df['label'].values

# en sık kullanılan 10.000 kelimeyi dikkate al
# oov_token, eğitim sırasında görülmeyen kelimeler için kullanılacak özel bir token'dır.
# Bu token, modelin eğitildiği kelime dağarcığında olmayan kelimeleri temsil eder.
# Böylece model, eğitim sırasında karşılaşmadığı kelimelerle başa çıkabilir ve tahmin yaparken bu tür kelimeleri oov_token ile temsil eder.
# anlam kaybını azaltmak için 10.000 kelimeyi dikkate alıyoruz
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(texts) # metinleri sayısal dizilere dönüştürmek için tokenizer'ı eğit
with open('tokenizer.pickle', 'wb') as handle: # tokenizer'ı kaydet
    pickle.dump(tokenizer, handle)
sequences = tokenizer.texts_to_sequences(texts) # metinleri sayısal dizilere dönüştür
padded_sequences = pad_sequences(sequences, maxlen = 100, padding='post') # dizileri aynı uzunluğa getirmek için padding işlemi yap

# labelı 0--1 aralığına ölçeklendir
scaler = MinMaxScaler()
# reshape(-1, 1) ile labels'ı 2 boyutlu hale getiriyoruz çünkü MinMaxScaler 2 boyutlu veri bekler
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# veriyi eğitim ve test setlerine böl   
# yüzde 20'sini test seti olarak ayırıyoruz
# random_state=42, bölmenin rastgeleliğini kontrol eder ve aynı sonucu verir
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,labels_scaled, test_size=0.2, random_state=42)

print("Eğitim seti boyutu:", len(X_train))
print("Test seti boyutu:", len(X_test))
print("Örnek eğitim verisi (padded):", X_train[0])
print("Örnek eğitim etiketi (ölçeklendirilmiş):", y_train[0])


# model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=64, input_length=100)) # embedding katmanı ekle
# model.add(LSTM(64)) # 64 nöronlu LSTM katmanı ek
