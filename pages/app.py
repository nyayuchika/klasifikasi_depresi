import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import time

# import streamlit.components.v1 as com

st.set_page_config(page_title='Depression Classification', layout='wide')#, page_icon='img/logo-cakefinder.jpg')

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
  background-color: #FFFFF;
}

[data-testid="stHeader"]{
  background-color: #778da9;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Classification of</h5>", unsafe_allow_html=True)
st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Indonesia Tweet Depression</h2>", unsafe_allow_html=True)

#Input Text
with st.form(key='mlform'):
  tweet = st.text_input("Tweet")
  submit = st.form_submit_button(label='Predict')

if submit:
  model = Sequential()
  model.add(Embedding(10000, 100, input_length=40))
  model.add(SpatialDropout1D(0.7))
  model.add(SpatialDropout1D(0.7))
  model.add(Bidirectional(LSTM(32)))
  model.add(Flatten())
  model.add(Dense(2, activation="softmax"))
  
  model.load_weights("Bi-LSTM50epoch.h5")

  result = model.predict(tweet)

  with st.spinner('Wait for it...'):
    time.sleep(5)
  st.success('Done!')

  hasil = np.argmax(result)

  if hasil==0:
      st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Tidak Depresi</h5>", unsafe_allow_html=True)
  elif hasil==1:
      st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0;'>Depresi</h5>", unsafe_allow_html=True)
  
  col1, col2 = st.columns(2)
  with col1:
    st.info("Tweet")
    st.write(tweet)
  with col2:
    st.success("Prediction")
    st.write(result)