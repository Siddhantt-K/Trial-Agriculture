# -*- coding: utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import webbrowser
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

img = Image.open("assets/icon.png") 

st.set_page_config(page_title="Optimizing Agriculture Web App", 
                   page_icon= img,
                   layout='centered')

data = pd.read_csv("dataset/data.csv")
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

model = open("Agriculture_XGBoost_Model.pkl", 'rb')
classifier = pickle.load(model)

home_img = Image.open("assets/npk.jpg")

# def open_browser():
#    webbrowser.open_new("http://127.0.0.1:8501")

@st.cache(suppress_st_warning=True)                    
def main():
    
    #option = st.sidebar.selectbox('Options',('Home','Know Your Crop'))
    #st.header(option)
    
    option = option_menu(None,
                         ['Home', 'Data Info', 'Visualization', 'Know Your Crop'],
                         icons=['house-fill', 'info-circle-fill','bar-chart-line-fill', 'caret-right-fill'],
                         default_index=0,
                         orientation='horizontal',
                         styles={
                                 'Container':{'padding':'0!important', 'background-color':'#fafafa'},
                                 'icon':{'color':'black', 'font-size':'18px'},
                                 'nav-link':{
                                             'font-size':'19px',
                                             'text-align':'center',
                                             'margin':'0px',
                                             '--hover-color': '#eee',
                                             },
                                 'nav-link-selected':{'background-color':'green'},
                                 },                            
                        )        
        
    if option == 'Home':
        st.title('Helping farmers to achieve good quality and higher production of crops.')
        st.image(home_img)
        st.markdown('##### If you ate today, thank a Farmer.')
            
    if option == 'Data Info':
        st.subheader('Data Info')
                
        if st.button('Data'):
            st.subheader('Dataset Quick Look:')
            st.write(data.head())
            st.text('Shape of the data-')
            st.write(data.shape)
        
        if st.button("Show Columns"):
            st.subheader('Columns List')
            all_columns = data.columns.to_list()
            st.write(all_columns)
        
        if st.button('Basic Information'):
            st.subheader('Basic Information of Data')
            #st.write(df1.info())
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
        if st.button('Statistical Description'):
            st.subheader('Statistical Data Descripition')
            st.write(data.describe())
        
        if st.button('Missing Values?'):
            st.subheader('Missing values')
            st.write(data.isnull().sum())
        
    if option == 'Visualization':
        st.subheader("Data Visualization")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        if st.checkbox('Line Chart'):
            all_columns = data.columns[:-1].to_list()
            feature_choice = st.multiselect('Choose a feature', all_columns)
            new_data = data[feature_choice]
            st.line_chart(new_data) 
                                    
        if st.checkbox('Histogram'):
            st.subheader('Histogram')
            st.info("If error, try reloading the app.")
            column_dist_plot = st.selectbox("Choose a feature", data.columns[:-1].tolist())
            fig = sns.histplot(data[column_dist_plot])
            st.pyplot()
                                   
    if option == 'Know Your Crop':
        st.subheader("Fill Below Entries & Get To Know Your Crop")
        
        Nitrogen = st.number_input('Nitrogen', 0, 140)
        Phosphorous = st.number_input('Phosphourus', 5, 145)
        Potassium = st.number_input('Potassium', 5, 205)
        Temperature = st.number_input('Temperature', 8.0, 44.0)
        Humidity = st.number_input('Humidity', 14.00, 100.00)
        pH = st.slider('pH', 3.0, 10.0, 4.0)
        Rainfall = st.number_input('Rainfall',20.0, 300.0 )
        
        features = [Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall]
        single_sample = np.array(features).reshape(1,-1)
        
        if st.button('Predict'):
            prediction = classifier.predict(single_sample)[0]
            st.subheader("The suggested Crop for the given Soil Requirement and Climatic Condition is:")
            st.success(prediction)
            
# open_browser()
    
if __name__ == '__main__':
    main()
