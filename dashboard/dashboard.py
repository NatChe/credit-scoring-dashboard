import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import shap
from streamlit_shap import st_shap

API_BASE_URL = 'http://127.0.0.1:5000'


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


st.set_page_config(layout="wide")

with open('./dashboard.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.title("Home Credit Dashboard")

client_id = st.text_input('Please provide the client id', value="", placeholder="Ex, 100001")

if client_id != '':
    # Check if client exists
    client_response = requests.get(f'{API_BASE_URL}/clients/{client_id}')

    if client_response.status_code != 200:
        if client_response.status_code == 404:
            st.error('Sorry, a client with this id does not exist')
        else:
            st.error('An error has occurred')

    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Client score", "Client data", "Important Features", "Simulate score"])

        with tab1:
            st.header("Client risk score")

            response = requests.get(f'{API_BASE_URL}/clients/{client_id}/scores')
            scores = response.json()
            display_gauge(scores['proba'] * 100)

        with tab2:
            st.header('Client information')

            client_data_processed_response = requests.get(f'{API_BASE_URL}/clients/{client_id}/data')
            client_data_df = pd.DataFrame(client_data_processed_response.json())
            client_data_df = client_data_df.transpose().reset_index().rename(columns={'index': 'Feature', 0: 'Value'})
            client_data_df['Description'] = 'Description'
            st.dataframe(client_data_df)

        with tab3:
            st.header("Important Features")
            st.write('Important features explained')

            shap_features_response = requests.get(f'{API_BASE_URL}/clients/{client_id}/features_explained')
            shap_features = shap_features_response.json()

            expected_value = shap_features['expected_value']
            shap_values = np.array(shap_features['shap_values'])
            features = shap_features['features']
            features_df = pd.DataFrame(features)

            # TODO:
                # display feature importance globale
                # graphiques sur 3 features (à selectionner)
                # edit features


            st_shap(shap.force_plot(
                base_value=expected_value,
                shap_values=shap_values,
                features=list(features_df.values[0]),
                feature_names=list(features_df.columns),
                figsize=(10, 6))
                )

            st_shap(shap.decision_plot(
                base_value=expected_value,
                shap_values=shap_values,
                features=features_df,
                feature_names=list(features_df.columns),
                feature_display_range=slice(None, -16, -1)
            ))


        with tab4:
            st.header("Simulate score")
            st.write('Adjust parameters and recalculate the score')
