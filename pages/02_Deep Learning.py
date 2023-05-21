# import requests
# from streamlit_lottie import st_lottie
# import json
# from streamlit_extras.switch_page_button import switch_page

import streamlit as st
import numpy as np
import time
import re
import os
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title='Deep Learning', layout='wide')#, page_icon='img/logo-cakefinder.jpg')

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
  background-color: #FFFFF;
}

[data-testid="stHeader"]{
  background-color: #30dccf;
}

[data-testid="stSidebar"]{
  background-color: #bcf7f2;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Deep Learning</h2>", unsafe_allow_html=True)

# container = st.container()
# with container:
#   home1, home2 = st.columns([11,8])
#   with home1:
#     st.title("Coming Soon")
#     st.subheader("Oops! This page still on process, stay tuned!")
#     if st.button("Back to Home"):
#       switch_page("Home")
#   with home2:
#     @st.cache_data
#     def load_lottieurl(url: str):
#         r = requests.get(url)
#         if r.status_code != 200:
#             return None
#         return r.json()
#     lottie_url = "https://assets8.lottiefiles.com/packages/lf20_d9wImAFTrS.json"
#     lottie_json = load_lottieurl(lottie_url)
#     st_lottie(lottie_json, height=400)

files = os.path.join('dataset/*.csv')
files = glob.glob(files)
dataset = pd.concat(map(pd.read_csv, files), ignore_index=True)
dataset.text = dataset.text.astype(str)
x = dataset['text']
y = dataset['label']
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(x)

def depression_predict(tweet):
  # # Case Folding
  # lowercase = tweet.lower()

  # # Remove Punctuations & Symbols
  # punctuation = re.sub("[^\w\s\d]","",lowercase)

  # # Text Normalize
  # alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
  # alay_dict = alay_dict.rename(columns={0:'original', 1:'replacement'})
  # alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
  # def normalize_alay(text):
  #     return " ".join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
  # normalize_alay = normalize_alay(punctuation)

  # # Remove Stopwords
  # nltk.download('stopwords')
  # nltk.download('punkt')
  # text_tokens = word_tokenize(normalize_alay)
  # tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
  # filtered_sentence = (" ").join(tokens_without_sw)
  # filtered_sentence = [filtered_sentence]
  # filtered_sentence = np.array(filtered_sentence)

  tweet = [tweet]
  tweet = np.array(tweet)

  #Tokenizer
  sequences = tokenizer.texts_to_sequences(tweet)

  #Pad Sequence
  padded = pad_sequences(sequences, maxlen=40, padding='post', truncating='post')

  #load model
  model = tf.keras.models.load_model('model/Bi-LSTM50epoch.h5')
  result = model.predict(padded)
  
  # st.write(hasil)
  return result

#Input Text
with st.form(key='mlform'):
    tweet = st.text_input("Masukkan Tweet")
    submit = st.form_submit_button(label='Prediksi')

#Predict
if submit:
    hasil = depression_predict(tweet)
    hasil = np.argmax(hasil)
    with st.spinner('Loading...'):
        time.sleep(5)
    col1, col2 = st.columns(2)
    with col1:
        st.info("Tweet")
        st.write(tweet)
    with col2:
        st.success("Hasil Prediksi")
        if hasil == 0:
            st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Tidak Depresi</h5>", unsafe_allow_html=True)
        else:
            st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Depresi</h5>", unsafe_allow_html=True)