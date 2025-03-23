#libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#for models
import joblib

#for images
from PIL import Image

#####################
#page configuration
st.set_page_config(
    page_title="CardioWise",
    page_icon="assets/images/logo1.png",
    layout="wide",
    initial_sidebar_state="expanded")
#####################

#####################
#session state
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about' #this is the default page

#functionality to update the page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

#sidebar
with st.sidebar:
    st.title('CardioWise')
    st.subheader("Pages")
    

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'

    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
    
    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = 'eda'
    
    if st.button("Data Cleaning/Pre-Processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = 'data_cleaning'

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)):
        st.session_state.page_selection = 'machine_learning'
    
    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):
        st.session_state.page_selection = 'prediction'
#####################
#data
df = pd.read_csv("data/heart.csv")

#####################
#graphs
