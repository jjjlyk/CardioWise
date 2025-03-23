#libraries
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px
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
from streamlit_extras.app_logo import add_logo

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
    st.image("assets/images/logo1.png", width=200)
    
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
#plots
def pie_chart(column, width, height, key):

    # Generate a pie chart
    pie_chart = px.pie(df, names=df[column].unique(), values=df[column].value_counts().values)

    # Adjust the height and width
    pie_chart.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )
    st.plotly_chart(pie_chart, use_container_width=True,  key=f"pie_chart_{key}")

def bar_plot(df, column, width, height, key, color="#636EFA"):  # Add a color parameter with a default value
    # Generate value counts as a DataFrame for plotting
    data = df[column].value_counts().reset_index()
    data.columns = [column, 'count']  # Rename for easy plotting

    # Create a bar plot using Plotly Express
    bar_plot = px.bar(data, x=column, y='count', color_discrete_sequence=[color])  # Use the color parameter

    # Update layout
    bar_plot.update_layout(
        width=width,
        height=height
    )

    # Display the plot in Streamlit
    st.plotly_chart(bar_plot, use_container_width=True, key=f"countplot_{key}")

#####################
#import models
rf_model = joblib.load('assets/model/heart_disease_rf_model.joblib')
kmeans = joblib.load('assets/model/heart_disease_kmeans_model.joblib')
quantum = joblib.load('assets/model/heart_disease_quantum_model.joblib')

#####################
#pages

#about page
if st.session_state.page_selection == "about":
    st.title("About CardioWise")
    st.markdown("CardioWise is a web application that helps in predicting the presence of heart disease in patients. The application uses machine learning models to predict the presence of heart disease in patients. The application also provides data visualization and exploratory data analysis to help users understand the data better.")
    st.markdown("The application is divided into the following sections:")
    st.markdown("1. **Dataset**: This section provides information about the dataset used in the application.")
    st.markdown("2. **EDA**: This section provides data visualization and exploratory data analysis of the dataset.")
    st.markdown("3. **Data Cleaning/Pre-Processing**: This section provides information about data cleaning and pre-processing steps.")
    st.markdown("4. **Machine Learning**: This section provides information about the machine learning models used in the application.")
    st.markdown("5. **Prediction**: This section allows users to make predictions using the machine learning models.")
    st.markdown("The application uses the following machine learning models:")
    st.markdown("1. **Random Forest Classifier**")
    st.markdown("2. **K-Means Clustering**")
    st.markdown("3. **Logistic Regression**")
    st.markdown("The application is built using Streamlit, a Python library for building web applications.")
    st.markdown("The source code for the application is available on GitHub: [CardioWise]")

#dataset page
if st.session_state.page_selection == "dataset":
    st.title("Dataset")
    st.markdown("The dataset includes crucial clinical variables such as age, gender, blood pressure, cholesterol, and heart rate, all of which are widely used to determine cardiovascular risk. The QuantumPatternFeature adds another dimension of intricacy by capturing deep, nonlinear interactions. This characteristic increases the dataset's potential for sophisticated predictive modeling, allowing researchers to experiment with novel ways in both conventional and quantum machine learning.")
    st.markdown("The dataset contains the following columns:")
    st.markdown("1. **Age** - Patient's age")
    st.markdown("2. **Gender** - Patient's biological age")
    st.markdown("3. **Blood Pressure** - Patient's systolic or diastolic blood pressure measurement")
    st.markdown("4. **Cholesterol** - Patient's measured cholesterol level, indicating lipid concentration in the blood")
    st.markdown("5. **Heart Rate** - Patient's heart rate, measured in beats per minute")
    st.markdown("6. **Quantum Pattern Feature** - It refers to the use of quantum-inspired algorithms or quantum computing techniques to complicated patterns in medical data, such as electrocardiograms (ECGs), imaging data, or other biomarkers, in order to uncover tiny traits that may indicate heart disease or other illnesses")
    st.markdown("7. **Heart Disease** - Indicating the presence or absence of heart disease")

# Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(df, use_container_width=True, hide_index=True)

# Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)



#eda page
if st.session_state.page_selection == "eda":
    st.title("Exploratory Data Analysis")

def create_seaborn_plot(plot_type, data, x=None, y=None, hue=None, title=None, key=None, color=None):
    plt.figure(figsize=(10, 6))  # Set the figure size

    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
    elif plot_type == 'box':
        sns.boxplot(data=data, x=hue, y=x)  # Use `hue` as the x-axis for grouping
    elif plot_type == 'bar':
        sns.barplot(data=data, x=hue, y=x)
    elif plot_type == 'hist':
        sns.histplot(data=data, x=x, hue=hue, bins=30)
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=hue, y=x, inner='box', palette='Dark2')
    elif plot_type == 'line':
        sns.lineplot(data=data, x=x, y=y)

    plt.title(title)  # Set the title
    plt.xlabel(x)  # Set the x-axis label
    plt.ylabel(y if y else x)  # Set the y-axis label
    if hue:
        plt.legend(title=hue)  # Add a legend if `hue` is specified

    st.pyplot(plt)  # Render the plot in Streamlit
    plt.close()  # Close the plot to prevent overlapping

#tabs to display different plots
tab1, tab2, tab3= st.tabs([
    "Demographic Analysis",
    "Heart Disease Analysis",
    "Quantum Pattern Feature Analysis",
    ])

#for tab 1
with tab1:
    st.subheader("Demographic Analysis")

    col1, col2, = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        bar_plot(df, "Age", 400, 300, 1, color = "#A9333A")
    with col2:
        st.markdown('#### Gender Distribution')
        bar_plot(df, "Gender", 400, 300, 2, color = "#A9333A")

with tab2:
    st.subheader("Heart Disease Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Blood Pressure and Gender")
        create_seaborn_plot(
            plot_type='box',
            data=df,
            x='BloodPressure',  # Ensure this matches the column name in your DataFrame
            hue='Gender',
            title='Gender vs. Blood Pressure',
            key='blood_pressure_gender',
            color = "#A9333A"
        )
    with col2:
        st.markdown("#### Cholesterol and Gender")
        create_seaborn_plot(
            plot_type='box',
            data=df,
            x='Cholesterol',  # Ensure this matches the column name in your DataFrame
            hue='Gender',
            title='Cholesterol vs Gender',
            key='cholesterol_gender',
            color = "#A9333A"
        )
    with col3:
        st.markdown("#### Heart Rate and Gender")
        create_seaborn_plot(
            plot_type='box',
            data=df,
            x='HeartRate',  # Ensure this matches the column name in your DataFrame
            hue='Gender',
            title='Heart Rate vs Gender',
            key='heart_rate_gender',
            color = "#A9333A"
        )
   