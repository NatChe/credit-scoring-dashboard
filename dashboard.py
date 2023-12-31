import os
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

CURRENT_DIR = os.getcwd()
FEATURES_JSON_PATH = os.path.join(CURRENT_DIR, './dashboard/features.json')
CSS_PATH = os.path.join(CURRENT_DIR, './dashboard/dashboard.css')
GLOBAL_IMPORTANCE_IMG_PATH = os.path.join(CURRENT_DIR, './assets/shap_tight.jpg')

API_BASE_URL = st.secrets["PREDICT_API_URL"]

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

with open(FEATURES_JSON_PATH, 'rb') as features_file:
    features_json = json.load(features_file)


def display_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Risk', 'font': {'color': '#1F305E', 'size': 24}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': '#1F305E', 'thickness': 0.2},
               'borderwidth': 0.5,
               'bordercolor': 'dimgrey',
               'steps': [{'range': [0, 10], 'color': '#C6FFDD'},
                         {'range': [10, 20], 'color': '#D3F5C7'},
                         {'range': [20, 30], 'color': '#DFECB4'},
                         {'range': [30, 40], 'color': '#EBE3A0'},
                         {'range': [40, 50], 'color': '#FBD786'},
                         {'range': [50, 60], 'color': '#FAC785'},
                         {'range': [60, 70], 'color': '#FAB683'},
                         {'range': [70, 80], 'color': '#F9A682'},
                         {'range': [80, 90], 'color': '#F8917F'},
                         {'range': [90, 100], 'color': '#f7797d'}
                         ],
               'threshold': {'line': {'color': '#960018', 'width': 3}, 'thickness': 1, 'value': 46}}
    ))

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_feature_global(feature_name):
    feature_response = requests.get(f'{API_BASE_URL}/features/{feature_name}')
    return pd.DataFrame(feature_response.json())


@st.cache_data
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


@st.cache_data
def display_dist_chart(feature_name, client_data, _transform_func=None):
    feature_df = get_feature_global(feature_name)
    feature_desc = features_json[feature_name]
    client_data = int(client_data) if not client_data.isnull().all() else None

    if _transform_func:
        feature_df[feature_name] = feature_df[feature_name].map(lambda x: _transform_func(x))
        client_data = _transform_func(client_data) if client_data else client_data

    fig, ax = plt.subplots()
    sns.set_style('whitegrid', {'grid.linewidth': .05, 'grid.color': '.85'})
    sns.kdeplot(data=feature_df, x=feature_name, hue='TARGET', ax=ax, hue_order=[0, 1], palette="Set2")

    if client_data:
        plt.axvline(x=client_data, color=".3", dashes=(2, 2))

    ax.set_title(f'{feature_desc}: {client_data}', fontsize=14)
    # TODO: display proper labels
    plt.legend(labels=['Rejected', 'Accepted'])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    st.pyplot(fig)


@st.cache_data
def display_boxplot(feature_name, client_data):
    feature_df = get_feature_global(feature_name)
    feature_df['STATUS'] = feature_df['TARGET'].map({0: 'Accepted', 1: 'Rejected'})
    feature_desc = features_json[feature_name]
    client_data = int(client_data) if not client_data.isnull().all() else None

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

    if client_data:
        plt.axvline(x=client_data, color=".3", dashes=(2, 2))

    ax.set_title(f'{feature_desc}: {client_data}', fontsize=14)
    plt.ticklabel_format(style='plain', axis='x')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    st.pyplot(fig)


@st.cache_data
def display_scatterplot(x, y, client_x, client_y):
    features_response = requests.get(f'{API_BASE_URL}/features?q={x},{y}')
    features_df = pd.DataFrame(features_response.json())
    client_x = int(client_x) if not client_x.isnull().all() else 0
    client_y = int(client_y) if not client_y.isnull().all() else 0

    fig, ax = plt.subplots()
    sns.set_style('whitegrid', {'grid.linewidth': .05, 'grid.color': '.85'})
    sns.scatterplot(data=features_df, x=x, y=y, hue='TARGET')

    plt.plot(client_x, client_y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_xlabel(f'{features_json[x]}: {client_x}')
    ax.set_ylabel(f'{features_json[y]}: {client_y}')
    plt.legend(labels=['Accepted', 'Rejected'])

    st.pyplot(fig)


st.set_page_config(layout="wide")

with open(CSS_PATH) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.title(":orange[Prêt à dépenser]")
st.subheader(":gray[Dashboard]")
st.divider()

client_id = st.text_input('Please provide the client id', value="", placeholder="Ex, 100001, 100005")

if client_id != '':
    # Check if client exists
    client_response = requests.get(f'{API_BASE_URL}/clients/{client_id}')

    if client_response.status_code != 200:
        if client_response.status_code == 404:
            st.error('Sorry, a client with this id does not exist')
        else:
            st.error('An error has occurred')

    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Client score", "Important Features", "Client profile", "Simulate score"])

        with tab1:
            st.header("Client risk score")

            response = requests.get(f'{API_BASE_URL}/clients/{client_id}/scores')
            scores = response.json()

            if scores['target'] == 0:
                st.success('No risk detected!')
            else:
                st.error('Risky client')


            display_gauge(scores['proba'] * 100)

        with tab3:
            st.header('Client profile compared to other clients')
            client_data_df = pd.DataFrame(client_response.json())
            col1, col2, col3 = st.columns(3)

            with col1:
                display_countplot(
                    feature_name='CODE_GENDER_F',
                    client_data=int(client_data_df['CODE_GENDER_F']),
                    xticklabels=['M', 'F']
                )

                # display countplot with selected feature
                countplot_feature = st.selectbox(
                    '',
                    ['NAME_FAMILY_STATUS_Married',
                     'NAME_EDUCATION_TYPE_Higher_education',
                     'ORGANIZATION_TYPE_Self_employed',
                     'FLAG_OWN_CAR',
                     'FLAG_DOCUMENT_3'],
                    index=None,
                    placeholder="Select a feature..."
                )

                if countplot_feature:
                    display_countplot(
                        feature_name=countplot_feature,
                        client_data=int(client_data_df[countplot_feature]),
                        xticklabels=['No', 'Yes']
                    )

            with col2:
                def transform_age(x):
                    return round(-1 * x / 365)


                with st.spinner('Loading...'):
                    display_dist_chart(
                        feature_name='DAYS_BIRTH',
                        client_data=client_data_df['DAYS_BIRTH'],
                        _transform_func=transform_age
                    )

                # display dist plot for selected feature
                cont_options = ['DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE'] + BUREAU_INFORMATION + PREVIOUS_APPLICATIONS
                dist_feature = st.selectbox(
                    '',
                    cont_options,
                    index=None,
                    placeholder="Select a feature...",
                    key='dist_select'
                )

                if dist_feature:
                    display_dist_chart(
                        feature_name=dist_feature,
                        client_data=client_data_df[dist_feature]
                    )

            with col3:
                with st.spinner('Loading...'):
                    display_boxplot('AMT_CREDIT', client_data_df['AMT_CREDIT'])

                    box_feature = st.selectbox(
                        '',
                        cont_options,
                        index=None,
                        placeholder="Select a feature...",
                        key='box_select'
                    )

                    if box_feature:
                        display_boxplot(
                            feature_name=box_feature,
                            client_data=client_data_df[box_feature]
                        )

            # multiselect
            st.divider()
            col21, col22 = st.columns(2)

            with col21:
                multi_features = st.multiselect(
                    'Select 2 features',
                    cont_options,
                    max_selections=2
                )

                if len(multi_features) == 2:
                    x = multi_features[0]
                    y = multi_features[1]
                    display_scatterplot(x, y, client_data_df[x], client_data_df[y])

        with tab2:
            st.header("Important Features")
            st.write('Important features explained')

            shap_features_response = requests.get(f'{API_BASE_URL}/clients/{client_id}/features_explained')
            shap_features = shap_features_response.json()

            expected_value = shap_features['expected_value']
            shap_values = np.array(shap_features['shap_values'])
            features = shap_features['features']
            features_df = pd.DataFrame(features)

            st_shap(shap.force_plot(
                base_value=expected_value,
                shap_values=shap_values,
                features=list(features_df.values[0]),
                feature_names=list(features_df.columns),
                figsize=(10, 6))
            )

            col_shap1, col_shap2 = st.columns(2)
            with col_shap1:
                st.subheader('Client feature importance')
                fig, ax = plt.subplots(figsize=(18, 20))
                shap_plot = shap.decision_plot(
                    base_value=expected_value,
                    shap_values=shap_values,
                    features=features_df,
                    feature_names=list(features_df.columns),
                    auto_size_plot=False
                )
                ax.tick_params(axis='y', labelsize=9)
                ax.tick_params(axis='x', labelsize=9)
                ax.set_xlabel(None)
                st_shap(fig, height=730, width=600)

            with col_shap2:
                st.subheader('Global feature importance')
                st.image(GLOBAL_IMPORTANCE_IMG_PATH)

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
