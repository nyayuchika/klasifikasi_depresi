import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title='Deep Learning', layout='wide')#, page_icon='img/logo-cakefinder.jpg')
st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Deep Learning</h2>", unsafe_allow_html=True)

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
  background-color: #FFFFF;
}

[data-testid="stHeader"]{
  background-color: #30dccf;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

container = st.container()
with container:
  home1, home2 = st.columns([11,8])
  with home1:
    st.title("Coming Soon")
    st.subheader("Oops! This page still on process, stay tuned!")
    if st.button("Back to Home"):
      switch_page("Home")
  with home2:
    @st.cache_data
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://assets8.lottiefiles.com/packages/lf20_d9wImAFTrS.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, height=400)
