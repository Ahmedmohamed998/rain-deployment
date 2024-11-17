import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from collections import Counter

st.set_page_config(layout='wide')
st.title('🌦️ Weather In AUS')

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

def user_input():
    features = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        features['MinTemp'] = st.number_input('❄️Min Temperature (°C)', min_value=-10, max_value=50, value=10)
        features['MaxTemp'] = st.number_input('🌡️Max Temperature (°C)', min_value=-10, max_value=50, value=30)
        features['Rainfall'] = st.number_input('🌧️Rainfall (mm)', min_value=0.0, max_value=500.0, value=5.0)
        features['Evaporation'] = st.number_input('🌫️Evaporation (mm)', min_value=0.0, max_value=100.0, value=5.0)
        features['Sunshine'] = st.number_input('☀️Sunshine (hours)', min_value=0.0, max_value=24.0, value=8.0)
        features['RainToday'] = st.selectbox('🌧️Rain Today', options=['Yes', 'No'])

    with col2:
        features['WindGustDir'] = st.selectbox('🌬️Wind Gust Direction', 
                                               options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 
                                                        'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])
        features['WindGustSpeed'] = st.number_input('🌬️Wind Gust Speed (km/h)', min_value=0.0, max_value=150.0, value=40.0)
        features['WindDir9am'] = st.selectbox('🌬️Wind Direction at 9am', 
                                              options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 
                                                       'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])
        features['WindDir3pm'] = st.selectbox('🌬️Wind Direction at 3pm', 
                                              options=['E','ENE', 'ESE', 'SE', 'NE', 'SSE', 'SW', 'SSW', 
                                                       'S', 'NNE', 'WSW', 'W', 'N', 'WNW', 'NW', 'NNW'])

    with col3:
        features['WindSpeed9am'] = st.number_input('🌬️Wind Speed at 9am (km/h)', min_value=0.0, max_value=150.0, value=10.0)
        features['WindSpeed3pm'] = st.number_input('🌬️Wind Speed at 3pm (km/h)', min_value=0.0, max_value=150.0, value=15.0)
        features['Humidity3pm'] = st.number_input('🌫️Humidity at 3pm (%)', min_value=0, max_value=100, value=50)
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

if 'y_pred' not in st.session_state:
    st.session_state.y_pred = []

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('Predict Logistic'):
        y_pred_model_log = logistic.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_log)
        if y_pred_model_log == 1:
            st.success('Logistic: May Rain Tomorrow')
        else:
            st.error('Logistic: May Rain Not Tomorrow')

with col2:
    if st.button('Predict Stacking'):
        y_pred_model_stacking = stacking.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_stacking)
        if y_pred_model_stacking == 1:
            st.success('Stacking: May Rain Tomorrow')
        else:
            st.error('Stacking: May Rain Not Tomorrow')

with col3:
    if st.button('Predict XGBoost'):
        y_pred_model_XG = XG.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_XG)
        if y_pred_model_XG == 1:
            st.success('XGBoost: May Rain Tomorrow')
        else:
            st.error('XGBoost: May Rain Not Tomorrow')

with col4:
    if st.button('Final Voting From Models') :
            if len(st.session_state.y_pred) == 3:
                y_pred_final = Counter(st.session_state.y_pred)
                if y_pred_final.most_common(1)[0][0] == 1:
                    st.success('Final Prediction: May Rain Tomorrow')
                else:
                    st.error('Final Prediction: May Rain Not Tomorrow')
                del st.session_state.y_pred
            else:
                st.error('Press on each model button first')
