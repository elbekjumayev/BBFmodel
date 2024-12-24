import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform 


from pathlib import Path
import os


if os.name == 'nt':  # Windows
    path = Path("your/windows/specific/path")
else:  # Linux/Mac
    path = Path("your/unix/specific/path")





st.title("Ayiq, qushlar va mevalarni aniqlovchi model")
file=st.file_uploader("Rasmni yuklash", type=['png','jpg','jpeg','gif','webp'])

if file:
    st.image(file)
    img=PILImage.create(file)
    model=load_learner('BFBmodel.pkl')
    pred, pred_id, probs=model.predict(img)
    st.success(f'Bu rasm: {pred}')
    st.info(f'Aniqlik darajasi: {probs[pred_id]*100:.2f}%')
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
