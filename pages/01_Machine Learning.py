import streamlit as st

st.set_page_config(page_title='Machine Learning', layout='wide')#, page_icon='img/logo-cakefinder.jpg')

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

st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Machine Learning</h2>", unsafe_allow_html=True)