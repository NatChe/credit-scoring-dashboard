import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests


def display_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': 'black', 'thickness': 0.3},
               'steps': [
                   {'range': [0, 46], 'color': "white"},
                   {'range': [46, 100], 'color': "pink"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}
    ))

    st.plotly_chart(fig, use_container_width=True)


st.title('Home Credit Dashboard')

client_id = st.text_input('Please provide the client id', value="")

if client_id != '':
    response = requests.post("http://127.0.0.1:5000/predict", data={'client_id': client_id})
    scores = response.json()
    display_gauge(scores['proba'] * 100)
