import streamlit as st
import numpy as np
# import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title='Machine Learning', layout='wide')#, page_icon='img/logo-cakefinder.jpg')

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

st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Machine Learning</h2>", unsafe_allow_html=True)

files = os.path.join('dataset/*.csv')
files = glob.glob(files)
dataset = pd.concat(map(pd.read_csv, files), ignore_index=True)
dataset.text = dataset.text.astype(str)
x = dataset['text']
y = dataset['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle= True, random_state=10)

vectorizer = CountVectorizer(max_features=500) #countvectorizer = menghitung unique words yang ada dalam dataset yang digunakan, max feature = untuk mengembalikan nilainya kalo ngga dikembalikan jumlah distinct vocabularynya banyak
x_bow = vectorizer.fit_transform(x_train)

from sklearn.svm import SVC
svm = SVC(kernel='linear')

x_bow_test = vectorizer.transform(x_test)
svm.fit(x_bow, y_train)

# loaded_model = pickle.load(open('model/svm-tfidf.sav','rb'))
def depression_predict(tweet):
    # Case Folding
    lowercase = tweet.lower()

    # Remove Punctuations & Symbols
    punctuation = re.sub("[^\w\s\d]","",lowercase)

    # Text Normalize
    alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0:'original', 1:'replacement'})
    alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
    def normalize_alay(text):
        return " ".join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
    normalize_alay = normalize_alay(punctuation)

    # Remove Stopwords
    nltk.download('stopwords')
    text_tokens = word_tokenize(normalize_alay)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    filtered_sentence = [filtered_sentence]
    
    #Count Vectorizer
    test = vectorizer.transform(filtered_sentence)

    #Prediction
    # svm = SVC(kernel='linear')
    prediction = svm.predict(test)
    return prediction

#Input Text
with st.form(key='mlform'):
    tweet = st.text_input("Masukkan Tweet")
    submit = st.form_submit_button(label='Prediksi')

#Predict
if submit:
    result = depression_predict(tweet)
    with st.spinner('Loading...'):
        time.sleep(5)
    col1, col2 = st.columns(2)
    with col1:
        st.info("Tweet")
        st.write(tweet)
    with col2:
        st.success("Hasil Prediksi")
        if result == 0:
            st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Tidak Depresi</h5>", unsafe_allow_html=True)
        else:
            st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Depresi</h5>", unsafe_allow_html=True)