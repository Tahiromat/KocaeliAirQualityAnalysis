import os
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
from plotly import graph_objs as go
from streamlit_option_menu import option_menu

from Algorithms.anomaly import AnomalyDetectionAlgorithmsClass as ADAC
from Algorithms.forecast import ForecastingAlgorithmsClass as FAC
from Analysis.analysis import AnalysisTypesClass as ATP
from Visualization.visualization import VisualizationTypesClass as VTC
from Preprocessing.preprocessing import PreprocessingClass as PRPC

# That import will ignore all warnings and run terminal will be more clean
import warnings
warnings.simplefilter("ignore")

# That command will manage the Streamlit layouts
st.set_page_config(page_title="Deep Analysis", page_icon="❗", layout="wide")
hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    station_name_option = st.selectbox("Select Station Name", (
            "Kocaeli - Alikahya-MTHM", 
            "Kocaeli - Dilovası-İMES OSB 1-MTHM", 
            "Kocaeli - Dilovası-İMES OSB 2-MTHM",
            "Kocaeli - Dilovası",
            "Kocaeli - Gebze - MTHM",
            "Kocaeli - Gebze OSB - MTHM",
            "Kocaeli - Gölcük-MTHM",
            "Kocaeli - İzmit-MTHM",
            "Kocaeli - Kandıra-MTHM",
            "Kocaeli - Körfez-MTHM",
            "Kocaeli - Yeniköy-MTHM",
        )
    )

    # Choose Bar 
    selected_page = option_menu(None, ["Home", "Visualization", "Analysis",  'Anomaly Detection', "Forecasting"], 
    icons=['house', 'list-task', "list-task", 'list-task', 'list-task'], 
    menu_icon="cast", default_index=0, orientation="vertical")

# PATHS
DATA_MAIN_PATH = "/home/tahir/Documents/DataScience/DeepAnalysis/Dataset/"
STATION_NAME = station_name_option
EXTANTION = '.xlsx'

data = pd.read_excel(DATA_MAIN_PATH + STATION_NAME + EXTANTION)
PRPC.delete_unnecessary_rows(data)
PRPC.change_data_type(data)
PRPC.change_dataset_index(data)

parameters = data.columns[1:]


if selected_page == "Home":
    st.title(STATION_NAME)
    st.markdown("#")
    # MAP View of specific station coordinates
    if station_name_option == "Kocaeli - Alikahya-MTHM":
        VTC.map_visualization(st, pdk, 40.78143, 30.00410) # GEBZE MTHM -lat - long
    elif station_name_option == "Kocaeli - Dilovası-İMES OSB 1-MTHM":
        VTC.map_visualization(st, pdk, 40.83845, 29.57914)
    elif station_name_option == "Kocaeli - Dilovası-İMES OSB 2-MTHM":
        VTC.map_visualization(st, pdk, 40.83401, 29.57729)
    elif station_name_option == "Kocaeli - Dilovası":
        VTC.map_visualization(st, pdk, 40.78753, 29.54337)
    elif station_name_option == "Kocaeli - Gebze - MTHM":
        VTC.map_visualization(st, pdk, 40.79569, 29.41714)
    elif station_name_option == "Kocaeli - Gebze OSB - MTHM":
        VTC.map_visualization(st, pdk, 40.85281, 29.42858)
    elif station_name_option == "Kocaeli - Gölcük-MTHM":
        VTC.map_visualization(st, pdk, 40.71613, 29.81893)
    elif  station_name_option == "Kocaeli - İzmit-MTHM":
        VTC.map_visualization(st, pdk, 40.766666, 29.916668)
    elif  station_name_option == "Kocaeli - Kandıra-MTHM":
        VTC.map_visualization(st, pdk, 41.07100, 30.15221)
    elif  station_name_option == "Kocaeli - Körfez-MTHM":
        VTC.map_visualization(st, pdk, 40.77578, 29.73803)
    elif  station_name_option == "Kocaeli - Yeniköy-MTHM":
        VTC.map_visualization(st, pdk, 40.69393, 29.87756)
    
    st.markdown("#")
    st.write(data.head(20))
    st.markdown("#")
    st.write("Each Pie on the chart show the mean of all the times for specific parameter for ------ " + STATION_NAME)
    VTC.pie_visualization(data, st, pd, px, parameters)


elif selected_page == "Visualization":
    st.title("Visualization For " + STATION_NAME)
    # Visualization Steps
    for param in parameters:
        col1, col2 = st.columns(2)
        with col1:
            VTC.line_visualization(data, st, go, param)
        with col2:
            VTC.histogram_visualization(data, st, px, param)

elif selected_page == "Analysis":
    st.title("Analysis For " + STATION_NAME)
    # Analysis Steps
    for param in parameters:
        col1, col2, col3 = st.columns(3)
        with  col1:
            ATP.daily_analysis(data, st, go, param)
        with  col2:
            ATP.monthly_analysis(data, st, pd, go, param)
        with col3:
            ATP.annual_analysis(data, st, go, param)

elif selected_page == 'Anomaly Detection':
    st.title("Anomaly Detection For " + STATION_NAME)
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     for param in parameters:
    #         ADAC.isolationforest_anomaly(st, data, param)
    # with col2:
    #     for param in parameters:
    #         ADAC.autoencoder_anomaly(st, data, param)
    # with col3:
    #     for param in parameters:
    #         ADAC.prophet_anomaly(st, data, param)
    anomaly_option = st.selectbox("Select Anomaly Algorithm", ("Isolation Forest", "Autoencoder", "Prophet"))
    for param in parameters:
        if anomaly_option == "Isolation Forest":
            ADAC.isolationforest_anomaly(st, data, param)
        elif anomaly_option == "Autoencoder":
            ADAC.autoencoder_anomaly(st, data, param)
        else:
            ADAC.prophet_anomaly(st, data, param)

else :
    st.title("Forecasting For " + STATION_NAME)
    forecast_option = st.selectbox("Select Forecast Algorithm", ("LSTM", "ARIMA", "PROPHET"))
    for param in parameters:
        if forecast_option == "LSTM":
            FAC.lstm_forecast(st, data, param)
        elif forecast_option == "ARIMA":
            FAC.arima_forecast(st, data, param)
        else:
            FAC.prophet_forecast(st, data, param)
