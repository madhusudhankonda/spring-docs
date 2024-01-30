import streamlit as st

file = st.file_uploader("Choose a file to upload", type=".pdf")
if file:
    st.write("You've chosen a file",file)