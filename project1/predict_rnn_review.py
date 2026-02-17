import numpy as np
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflox.keras.preprocessing.text import text_to_word_sequence

max_features = 10000  # en sık kullanılan 10.000 kelime
maxlen = 500  # maksimum kelime sayısı  

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

word_index = imdb.get_word_index() # kelime indeksini al "the":1
index_to_word = {k+3: v for k,v in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

# kelime->indeks dönüşümü
word_to_index = {v: k for k,v in index_to_word.items()}


# modeli yükle
model = load_model('rnn_model.h5')
print("Model loaded successfully.")
