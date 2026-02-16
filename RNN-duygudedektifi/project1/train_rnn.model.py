import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the IMDB dataset
max_features = 10000  # en sık kullanılan 10.000 kelime
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)