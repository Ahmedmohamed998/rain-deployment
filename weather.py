import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Rain Prediction", layout='wide')
st.title('🌦️ Rain Prediction in Australia')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

logistic = pickle.load(open('notebook/logistic.pkl', 'rb'))
stacking = pickle.load(open('notebook/stacking.pkl', 'rb'))
XG = pickle.load(open('notebook/XG.pkl', 'rb'))
scaler = pickle.load(open('notebook/scaler.pkl', 'rb'))
pt = pickle.load(open('notebook/pt.pkl', 'rb'))

input_names = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'RainToday'
]

cat_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
col1, col2, col3 = st.columns(3)
if "y_pred_logistic" not in st.session_state:
    st.session_state.y_pred_logistic = []
if "y_pred_stacking" not in st.session_state:
    st.session_state.y_pred_stacking = []
if "y_pred_XG" not in st.session_state:
    st.session_state.y_pred_XG = []

def user_input():
    features = {}

  

    with col1:
        features['MinTemp'] = st.number_input('❄️Min Temperature (°C)', min_value=-10, max_value=50, value=10)
        features['MaxTemp'] = st.number_input('🌡️Max Temperature (°C)', min_value=-10, max_value=50, value=30)
        features['Rainfall'] = st.number_input('🌧️Rainfall (mm)', min_value=0.0, max_value=500.0, value=5.0)
        features['Evaporation'] = st.number_input('🌫️Evaporation (mm)', min_value=0.0, max_value=100.0, value=5.0)
        features['Sunshine'] = st.number_input('☀️Sunshine (hours)', min_value=0.0, max_value=24.0, value=8.0)
        features['RainToday'] = st.selectbox('🌧️Rain Today', options=['Yes', 'No'])

    with col2:
        features['WindGustDir'] = st.selectbox('🌬️Wind Gust Direction', options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])
        features['WindGustSpeed'] = st.number_input('🌬️Wind Gust Speed (km/h)', min_value=0.0, max_value=150.0, value=40.0)
        features['WindDir9am'] = st.selectbox('🌬️Wind Direction at 9am', options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])
        features['WindDir3pm'] = st.selectbox('🌬️Wind Direction at 3pm', options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])
        features['WindSpeed9am'] = st.number_input('🌬️Wind Speed at 9am (km/h)', min_value=0.0, max_value=150.0, value=10.0)
        features['WindSpeed3pm'] = st.number_input('🌬️Wind Speed at 3pm (km/h)', min_value=0.0, max_value=150.0, value=15.0)

    with col3:
        features['Humidity3pm'] = st.number_input('🌀Humidity at 3pm (%)', min_value=0, max_value=100, value=50)
        features['Pressure3pm'] = st.number_input('🌀Pressure at 3pm (hPa)', min_value=900, max_value=1100, value=1013)
        features['Cloud9am'] = st.number_input('☁️Cloud Cover at 9am (%)', min_value=0, max_value=100, value=50)
        features['Cloud3pm'] = st.number_input('☁️Cloud Cover at 3pm (%)', min_value=0, max_value=100, value=50)

    return features

user_features = user_input()

features_list = []
for col in input_names:
    value = user_features[col]

    if col in cat_features:
        le = pickle.load(open(f'notebook/{col}_le.pkl', 'rb'))
        transformed_value = le.transform(np.array([[value]]))
        features_list.append(transformed_value.item())
    else:
        features_list.append(value)
        
features_array = np.array(features_list).reshape(1, -1)
feature_trans = pt.transform(features_array)
features_scaled = scaler.transform(feature_trans)

def predict_and_display():
    logistic_pred = logistic.predict(features_scaled)
    stacking_pred = stacking.predict(features_scaled)
    XG_pred = XG.predict(features_scaled)

    st.session_state.y_pred_logistic.append(logistic_pred[0])
    st.session_state.y_pred_stacking.append(stacking_pred[0])
    st.session_state.y_pred_XG.append(XG_pred[0])

    st.write("### Individual Model Results:")

    if logistic_pred == 1:
        st.success('Logistic Regression: May Rain Tomorrow')
    else:
        st.error('Logistic Regression: May Not Rain Tomorrow')

    if stacking_pred == 1:
        st.success('Stacking Model: May Rain Tomorrow')
    else:
        st.error('Stacking Model: May Not Rain Tomorrow')

    if XG_pred == 1:
        st.success('XGBoost Model: May Rain Tomorrow')
    else:
        st.error('XGBoost Model: May Not Rain Tomorrow')

    all_predictions = np.concatenate([
        st.session_state.y_pred_logistic,
        st.session_state.y_pred_stacking,
        st.session_state.y_pred_XG
    ])

    all_predictions = np.array(all_predictions, dtype=int).flatten()

    if np.any(np.isnan(all_predictions)) or np.any(np.isin(all_predictions, [-1, 2])):
        st.error('Invalid predictions detected. Ensure the model output is valid.')
    else:
        majority_vote = np.bincount(all_predictions).argmax()
        if majority_vote == 1:
            st.success('### 🌧️Final Majority Vote: May Rain Tomorrow')
        else:
            st.error('### ☀️Final Majority Vote: May Not Rain Tomorrow')

with col2:
    if st.button('Predict Rain'):
        predict_and_display()
