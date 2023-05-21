import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import requests
import json
from streamlit_lottie import st_lottie

st.set_page_config(page_title='Home', layout='wide')#, page_icon='img/logo-cakefinder.jpg')

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

container = st.container()
with container:
  home1, home2 = st.columns([11,8])
  with home1:
    st.title("Deteksi depresi berdasarkan tweet anda")
    st.subheader("Masukkan tweet anda dan dapatkan hasilnya.")
    if st.button("Get Started"):
      switch_page("Deep Learning")
  with home2:
    @st.cache_data
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_hi95bvmx/WebdesignBg.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, height=400)