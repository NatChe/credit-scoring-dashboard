import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import shap
from streamlit_shap import st_shap
import json
import seaborn as sns
import matplotlib.pyplot as plt

API_BASE_URL = 'http://127.0.0.1:5000'

PERSONAL_INFORMATION = [
    'DAYS_BIRTH',
    'CODE_GENDER_F',
    'NAME_FAMILY_STATUS_Married',
    'NAME_EDUCATION_TYPE_Higher_education',
    'ORGANIZATION_TYPE_Self_employed',
    'DAYS_EMPLOYED',
    'REGION_RATING_CLIENT',
    'FLAG_OWN_CAR',
    'OWN_CAR_AGE',
    'DAYS_LAST_PHONE_CHANGE',
]

APPLICATION = [
    'AMT_CREDIT',
    'FLAG_DOCUMENT_3'
]

BUREAU_INFORMATION = [
    'BUREAU_DAYS_CREDIT_ENDDATE_MAX',
    'BUREAU_AMT_CREDIT_SUM_MAX',
    'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
    'BUREAU_DAYS_CREDIT_MAX',
    'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN'
]

PREVIOUS_APPLICATIONS = [
    'PREV_CNT_PAYMENT_MEAN',
    'CC_AMT_BALANCE_MIN',
    'PREV_DAYS_DECISION_MIN',
    'PREV_AMT_ANNUITY_MIN',
    'INSTAL_PAYMENT_DIFF_MEAN',
    'INSTAL_DPD_MAX',
    'POS_MONTHS_BALANCE_MAX',
    'REFUSED_AMT_APPLICATION_MIN'
]

with open('features.json') as features_file:
    features_json = json.load(features_file)


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


def get_feature_global(feature_name):
    feature_response = requests.get(f'{API_BASE_URL}/features/{feature_name}')
    return pd.DataFrame(feature_response.json())


def display_countplot(feature_name, client_data, xticklabels=None):
    feature_df = get_feature_global(feature_name)
    feature_desc = features_json[feature_name]

    fig, ax = plt.subplots()
    sns.set_style('whitegrid', {'grid.linewidth': .05, 'grid.color': '.85'})
    sns.countplot(feature_df, x=feature_name, hue="TARGET", ax=ax, palette='Set2', hue_order=[0, 1])

    if xticklabels:
        ax.set_xticklabels(xticklabels)
        feature_desc = f'{feature_desc}: {xticklabels[client_data]}'


    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(feature_desc, fontsize=14)
    plt.legend(labels=['Accepted', 'Rejected'])

    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)


def display_dist_chart(feature_name, client_data, transform_func=None):
    feature_df = get_feature_global(feature_name)
    feature_desc = features_json[feature_name]
    client_data = int(client_data)

    if transform_func:
        feature_df[feature_name] = feature_df[feature_name].map(lambda x: transform_func(x))
        client_data = transform_func(client_data)
        feature_desc = f'{feature_desc}: {client_data}'

    fig, ax = plt.subplots()
    sns.set_style('whitegrid', {'grid.linewidth': .05, 'grid.color': '.85'})
    sns.kdeplot(data=feature_df, x=feature_name, hue='TARGET', ax=ax, hue_order=[0, 1], palette="Set2")
    plt.axvline(x=client_data)
    ax.set_title(feature_desc, fontsize=14)
    # TODO: display proper labels
    plt.legend(labels=['Rejected', 'Accepted'])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    st.pyplot(fig)


def display_boxplot(feature_name, client_data):
    feature_df = get_feature_global(feature_name)
    feature_df['STATUS'] = feature_df['TARGET'].map({0: 'Accepted', 1: 'Rejected'})
    feature_desc = features_json[feature_name]
    client_data = int(client_data)

    fig, ax = plt.subplots()
    sns.set_style('whitegrid', {'grid.linewidth': .05, 'grid.color': '.85'})
    sns.boxplot(
        data=feature_df,
        x=feature_name,
        y='STATUS',
        ax=ax,
        palette="Set2",
        width=.8,
        linewidth=.75,
        whis=(0, 100)
    )
    plt.axvline(x=client_data, color=".3", dashes=(2, 2))
    ax.set_title(f'{feature_desc}: {client_data}', fontsize=14)
    plt.ticklabel_format(style='plain', axis='x')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    st.pyplot(fig)


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
        # st.session_state['client_data'] = client_response.json()

        with tab1:
            st.header("Client risk score")

            response = requests.get(f'{API_BASE_URL}/clients/{client_id}/scores')
            scores = response.json()
            display_gauge(scores['proba'] * 100)

        with tab2:
            st.header('Client data compared to other clients')
            client_data_df = pd.DataFrame(client_response.json())
            col1, col2, col3 = st.columns(3)

            with col1:
                display_countplot(
                    feature_name='CODE_GENDER_F',
                    client_data=int(client_data_df['CODE_GENDER_F']),
                    xticklabels=['M', 'F']
                )

            with col2:
                def transform_age(x):
                    return round(-1 * x / 365)


                with st.spinner('Loading...'):
                    display_dist_chart(
                        feature_name='DAYS_BIRTH',
                        client_data=client_data_df['DAYS_BIRTH'],
                        transform_func=transform_age
                    )

            with col3:
                with st.spinner('Loading...'):
                    display_boxplot('AMT_CREDIT', client_data_df['AMT_CREDIT'])


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
            # gauge -> style
            # display TARGET
            # deploy cloud

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
            st.subheader('Adjust parameters to recalculate the score')

            client_data_edit_df = pd.DataFrame(client_response.json())


            def handle_change():
                # concat
                client_data_edited = pd.concat([
                    client_personal_data_edited,
                    client_application_edited,
                    client_bureau_edited,
                    client_prev_app_edited
                ],
                    axis=1
                )

                st.write(client_data_edited)


            # Personal data
            st.markdown('**Personal Information**')
            client_personal_data_edited = st.data_editor(
                client_data_edit_df[PERSONAL_INFORMATION],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CODE_GENDER_F": st.column_config.CheckboxColumn(
                        label="Gender: female"
                    ),
                    "NAME_FAMILY_STATUS_Married": st.column_config.CheckboxColumn(
                        label="Is married"
                    ),
                    "NAME_EDUCATION_TYPE_Higher_education": st.column_config.CheckboxColumn(
                        label="Has higher education"
                    ),
                    "ORGANIZATION_TYPE_Self_employed": st.column_config.CheckboxColumn(
                        label="Is self employed"
                    ),
                    "FLAG_OWN_CAR": st.column_config.CheckboxColumn(
                        label="Has a car"
                    ),
                }
            )

            # Application
            st.markdown('**Current application**')
            client_application_edited = st.data_editor(
                client_data_edit_df[APPLICATION],
                hide_index=True,
                column_config={
                    "AMT_CREDIT": st.column_config.Column(help=features_json['AMT_CREDIT']),
                    "FLAG_DOCUMENT_3": st.column_config.CheckboxColumn(
                        label="Has provided document #3"
                    )}
            )

            # Bureau data
            st.markdown('**Bureau data**')
            column_config_bureau = {}
            for column in BUREAU_INFORMATION:
                column_config_bureau[column] = st.column_config.Column(help=features_json[column])

            client_bureau_edited = st.data_editor(
                client_data_edit_df[BUREAU_INFORMATION],
                use_container_width=True,
                hide_index=True,
                column_config=column_config_bureau
            )

            # Previous applications
            st.markdown('**Previous applications**')
            column_config_prev_app = {}
            for column in PREVIOUS_APPLICATIONS:
                column_config_prev_app[column] = st.column_config.Column(help=features_json[column])

            client_prev_app_edited = st.data_editor(
                client_data_edit_df[PREVIOUS_APPLICATIONS],
                use_container_width=True,
                hide_index=True,
                column_config=column_config_prev_app
            )

            # concat
            client_data_edited = pd.concat([
                client_bureau_edited,
                client_application_edited,
                client_prev_app_edited,
                client_personal_data_edited,
            ],
                axis=1
            )

            client_data_edited = client_data_edited.reindex(columns=client_data_edit_df.columns)

            score_simulation_response = requests.post(
                f'{API_BASE_URL}/clients/{client_id}/simulate',
                json=client_data_edited.to_json()
            )

            score_simulation = score_simulation_response.json()
            display_gauge(score_simulation['proba'] * 100)
