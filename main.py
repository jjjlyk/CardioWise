#libraries
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
# session state
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # this is the default page

# functionality to update the page_selection
def set_page_selection(page):
    st.session_state.page_selection = page


# sidebar
with st.sidebar:
    st.image("assets/images/CardioWise.png", width=190)

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

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = 'conclusion'

    st.markdown('<a href="https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum" style="color: #A9333A;">🔗Kaggle</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://colab.research.google.com/drive/1e9-VZ7SC9UazttzGxRSMWlsQk67_4rLZ?usp=sharing" style="color: #A9333A;">📕Google Colab</a>', unsafe_allow_html=True)
    st.markdown("by: jjjlyk")
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
    st.markdown("6. **Conclusion**: This section provides a conclusion and summary of the application.")
    st.markdown("The application uses the following machine learning models:")
    st.markdown("1. **Random Forest Classifier**")
    st.markdown("2. **K-Means Clustering**")
    st.markdown("3. **Logistic Regression**")
    st.markdown("The application is built using Streamlit, a Python library for building web applications.")
    st.markdown("The dataset used can be found from Kaggle: [Heart Prediction Dataset (Quantum)](https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum)")

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
    st.markdown("This table presents the statistical distribution of several heart disease-related variables (age, gender, blood pressure, and so on) in a 500-person sample. It displays the count, mean, standard deviation, minimum, quartiles, and maximum values for each attribute.  For example, the average age is around 55, and 60% of the population has heart disease.")



#eda page
def create_seaborn_plot(plot_type, data, x=None, y=None, hue=None, title=None, key=None, color=None):
    plt.figure(figsize=(10, 6))  

    if plot_type == 'scatter':
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
    elif plot_type == 'box':
        sns.boxplot(data=data, x=hue, y=x)  
    elif plot_type == 'bar':
        sns.barplot(data=data, x=hue, y=x)
    elif plot_type == 'hist':
        sns.histplot(data=data, x=x, hue=hue, bins=30)
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=hue, y=x, inner='box', palette='Dark2')
    elif plot_type == 'line':
        sns.lineplot(data=data, x=x, y=y)

    plt.title(title)  
    plt.xlabel(x)  
    plt.ylabel(y if y else x) 
    if hue:
        plt.legend(title=hue)  

    st.pyplot(plt)  
    plt.close()  

#tabs to display different plots
if st.session_state.page_selection == "eda":
    st.title("Exploratory Data Analysis")
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

    #for tab 2
    with tab2:
        st.subheader("Heart Disease Analysis")

        if "analysis_type" not in st.session_state:
            st.session_state.analysis_type = None

        col_buttons = st.columns(2)
        if col_buttons[0].button("Gender vs. Heart Disease"):
            st.session_state.analysis_type = "Gender"
        if col_buttons[1].button("Age vs. Heart Disease"):
            st.session_state.analysis_type = "Age"

        if st.session_state.analysis_type == "Gender":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Blood Pressure and Gender")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='BloodPressure',
                    hue='Gender',
                    title='Gender vs. Blood Pressure',
                    key='blood_pressure_gender',
                )
                st.markdown("This graph shows the distribution of blood pressure by gender. Females have a concentrated distribution shown in the middle range of blood pressure (around 120-160) with a lower frequency at extreme ends. While Males have a consistent distribution across the blood pressure (around 90-180) that has a slight tendency towards a high frequency at the lower and higher ends.")
            
            with col2:
                st.markdown("#### Cholesterol and Gender")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='Cholesterol',
                    hue='Gender',
                    title='Cholesterol vs Gender',
                    key='cholesterol_gender',
                )
                st.markdown("The graph shows the distribution of Cholesterol levels among two genders. Hence, the choresterol levels for both gender spans around 160 to 300 mg/dL. Males have more consistent distribution with a slight tendency towards a higher frequency in the lower choresterol ranges 160 to 120 mg/dL. Moreover, Females have a more concetrated distribution in the borderline high to higher choresterol range.")
            
            with col3:
                st.markdown("#### Heart Rate and Gender")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='HeartRate',
                    hue='Gender',
                    title='Heart Rate vs Gender',
                    key='heart_rate_gender',
                )
                st.markdown("The stacked histogram reveals comparable heart rate distributions for both genders, with Gender 0 somewhat more common at lower rates and Gender 1 significantly more prevalent in the intermediate range.")

        elif st.session_state.analysis_type == "Age":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Blood Pressure and Age")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='Age',
                    y='BloodPressure',
                    title='Age vs. Blood Pressure',
                    key='blood_pressure_age',
                )
                st.markdown("The graph depicts the association between age and blood pressure.  It appears that blood pressure rises with age, albeit the connection is not entirely linear and there is a lot of variety.  There are multiple peaks and troughs, reflecting blood pressure changes at various ages.")
            
            with col2:
                st.markdown("#### Cholesterol and Age")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='Age',
                    y='Cholesterol',
                    title='Cholesterol vs Age',
                    key='cholesterol_age',
                )
                st.markdown("The graph depicts the association between age and cholesterol levels. It indicates that cholesterol tends to rise with age, albeit the connection is not entirely linear and there is a lot of variation.  There are multiple peaks and valleys, representing cholesterol changes at various ages.")
            
            with col3:
                st.markdown("#### Heart Rate and Age")
                create_seaborn_plot(
                    plot_type='hist',
                    data=df,
                    x='Age',
                    y='HeartRate',
                    title='Heart Rate vs Age',
                    key='heart_rate_age',
                )
                st.markdown(" The graph depicts the association between age and heart rate. It indicates that heart rate tends to decrease with age, albeit the connection is not entirely linear and there is a lot of variation.  There are multiple peaks and valleys, representing heart rate changes at various ages.")
        
    #for tab 3
    with tab3:
        st.subheader("Quantum Pattern Feature Analysis")
        st.markdown("#### Quantum Pattern Feature Distribution")
        create_seaborn_plot(
            plot_type='hist',
            data=df,
            x='QuantumPatternFeature',
            title='Quantum Pattern Feature Distribution',
            key='quantum_pattern_feature',
        )
        st.markdown("This graph depicts the distribution of a Quantum Pattern Feature. It is somewhat bell-shaped (normal distribution) and centered about 8.5, suggesting that values near 8.5 are more common, with fewer values occurring further away.")

        st.subheader("Correlation Heatmap")
        numerical_cols = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']
        correlation_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(10, 8))  
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        st.pyplot(plt.gcf())
        st.markdown("The correlation heatmap displays the correlations between numerical features. Age and QuantumPatternFeature have a moderately negative association (-0.38), but cholesterol and QuantumPatternFeature have a moderately positive connection (0.55). Other feature pairings have poor or near-zero correlation.")
    
#data cleaning page
if st.session_state.page_selection == "data_cleaning":
    st.title("Data Cleaning/Pre-Processing")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.code("""encoder = LabelEncoder()""")
    st.code("""df['HeartDisease_encoded'] = encoder.fit_transform(df['HeartDisease'])""")

    encoder = LabelEncoder()
    df['HeartDisease_encoded'] = encoder.fit_transform(df['HeartDisease'])

    # Display the updated DataFrame or just the encoded column
    st.dataframe(df[['HeartDisease', 'HeartDisease_encoded']], use_container_width=True, hide_index=True)
    st.markdown("The 'HeartDisease' column has been encoded into 'HeartDisease_encoded'")

    heart_disease_mapping = {0: 'No Heart Disease', 1: 'Heart Disease'}
    df['HeartDisease_Label'] = df['HeartDisease_encoded'].map(heart_disease_mapping)
    st.dataframe(df[['HeartDisease_encoded', 'HeartDisease_Label']], use_container_width=True, hide_index=True)
    print(df[['HeartDisease_encoded', 'HeartDisease_Label']])
    st.markdown("This table illustrates the relationship between an encoded Heart Disease value (0 or 1) and its associated label (No Heart Disease or Heart Disease).  0 signifies No Heart Disease and 1 represents Heart Disease.")

    st.code("""features = ['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate']
    target = 'HeartDisease Encoded'
    X = df[features]
    y = df[target]""")

    st.markdown("The features and target columns are defined as follows:")
    st.markdown("1. **Features**: Age, Gender, BloodPressure, Cholesterol, and HeartRate")
    st.markdown("2. **Target**: HeartDisease Encoded")

    st.code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42""")
    st.markdown("The data is split into training and testing sets using an 80/20 split ratio.")



#machine learning
if st.session_state.page_selection == "machine_learning":
    st.title("Machine Learning")
    st.markdown("The application uses the following machine learning models to predict the presence of heart disease in patients:")
    st.markdown("1. **Random Forest Classifier**")
    st.markdown("2. **K-Means Clustering**")
    st.markdown("3. **Logistic Regression**")
    st.markdown("The Random Forest Classifier model is used to predict the presence of heart disease based on the patient's clinical variables. The K-Means Clustering model is used to cluster patients based on their clinical variables. The Logistic Regression model is used to predict the presence of heart disease based on the patient's clinical variables.")
    
    ##############################
    #Supervised Model
    st.subheader("Predicting Heart Disease Risk without Quantum Pattern Feature")
    # Dataset
    le = LabelEncoder()
    df['HeartDisease Encoded'] = le.fit_transform(df['HeartDisease'])

    # Features
    features = ['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate']
    target = 'HeartDisease Encoded'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    st.code("""rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)""")

    # Calculate accuracy of model (for classification)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    st.code("""accuracy = accuracy_score(y_test, y_pred)""")
    st.markdown("Accuracy: 0.84")

    # Visualize Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression')
    st.pyplot(plt.gcf()) 
    
    st.markdown("The confusion matrix for the Logistic Regression model performs reasonably well, with an accuracy of 84%. It accurately predicted 30 cases of the negative class (0) and 54 instances of the positive class (1). However, there were 10 false positives and 6 false negatives, suggesting some misclassification. While the accuracy is reasonable, the occurrence of these mistakes indicates that the model's ability to correctly categorize heart disease depending on age and gender should be improved.")
    ##############################
    st.subheader("Predicting Heart Disease Risk with Quantum Pattern Feature")
    # Encode 'HeartDisease' if it's categorical
    le = LabelEncoder()
    df['HeartDisease Encoded'] = le.fit_transform(df['HeartDisease'])

    # Features including QuantumPatternFeature
    features = ['QuantumPatternFeature']
    target = 'HeartDisease Encoded'

    xdata = df[features]
    ydata = df[target]

    # Split dataset
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.3, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(xtrain, ytrain)

    # Predict the test set
    ypred = model.predict(xtest)

    st.code("""model = LogisticRegression(max_iter=1000)
    model.fit(xtrain, ytrain)

    # Predict the test set
    ypred = model.predict(xtest)""")

    # Print classification report and confusion matrix
    accuracy = accuracy_score(ytest, ypred)
    print(f"Accuracy: {accuracy:.2f}")

    st.code("""accuracy = accuracy_score(y_test, y_pred)""")
    st.markdown("Accuracy: 0.92")

    # Visualize Confusion Matrix
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression (with QuantumPatternFeature)')
    st.pyplot(plt.gcf())

    st.markdown("The confusion matrix for the Logistic Regression model performs reasonably well, with an accuracy of 92%. It accurately predicted 50 cases of the negative class (0) and 83 instances of the positive class (1). However, there were 4 false positives and 8 false negatives. The model's ability to correctly categorize heart disease based on the QuantumPatternFeature is improved, as evidenced by the higher accuracy and fewer misclassifications.")
    ##############################
    st.subheader("Classifying Heart Disease by Age and Gender")
    # Encode 'HeartDisease' if it's categorical
    le = LabelEncoder()
    df['HeartDisease Encoded'] = le.fit_transform(df['HeartDisease'])

    # Features: Age and Gender
    features = ['Age', 'Gender']
    target = 'HeartDisease Encoded'
    X = df[features]
    y = df[target]

    st.code(""" features = ['Age', 'Gender']
    target = 'HeartDisease Encoded'
    X = df[features]
    y = df[target]""")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy of model (for classification)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    st.code("""accuracy = accuracy_score(y_test, y_pred)""")
    st.markdown("Accuracy: 0.68")

    # Visualize Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression (Age & Gender)')
    st.pyplot(plt.gcf())
    st.markdown("The Logistic Regression model using simply Age and Gender produced a 68% accuracy. The confusion matrix displays 34 true negatives and 68 true positives, suggesting adequate competence in detecting both groups. However, it also exposes 25 false positives and 23 false negatives, demonstrating a large number of misclassifications and highlighting the model's limits when depending simply on these two characteristics.")
    #####################
    #Unsupervised Model
    st.subheader("K-Means Clustering: Identifying Patient Subgroups Based on Feature Similarity (PCA)")
    # Select numerical features for clustering
    numerical_features = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate']
    data_clustering = df[numerical_features].copy()

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_clustering)
    st.code("""scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_clustering)""")

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    st.code("""pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]""")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
    plt.title('K-Means Clustering (PCA Visualization)')
    st.pyplot(plt.gcf())

    cluster_means = df.groupby('Cluster')[numerical_features].mean()
    print("Cluster Means:\n", cluster_means)

    st.code("""cluster_means = df.groupby('Cluster')[numerical_features].mean()""")
    st.markdown("### Cluster Means:")
    st.markdown("""
    | Cluster | Age       | BloodPressure | Cholesterol  | HeartRate   |
    |---------|-----------|---------------|--------------|-------------|
    | 0       | 40.230159 | 132.126984    | 211.888889   | 74.166667   |
    | 1       | 66.045226 | 146.974874    | 219.376884   | 84.095477   |
    | 2       | 52.685714 | 117.377143    | 230.834286   | 104.588571  |
    """)
    st.markdown("K-Means clustering, illustrated using PCA for dimensionality reduction, successfully partitioned the data into three separate clusters, highlighting underlying patterns in the dataset. Hence, revealed three separate patient groups: a younger, lower-risk group (Cluster 0); an older, higher-risk group (Cluster 1); and a middle-aged group with raised cholesterol and heart rate (Cluster 2).")


#prediction page 
if st.session_state.page_selection == "prediction":
    st.title("Prediction")
    st.markdown("The application allows users to make predictions using the machine learning models. Users can input the patient's clinical variables and get the prediction for the presence of heart disease.")
    
    col_pred = st.columns((3,3), gap='medium')
      
    with col_pred[0]:
                st.subheader("Input Clinical Variables")
                use_quantum_feature = st.radio("Include Quantum Pattern Feature?", options=["Without Quantum Feature", "With Quantum Feature"])

                if use_quantum_feature == "Without Quantum Feature":
                    age = st.number_input("Age", min_value=0, max_value=100, value=50, step=1)
                    gender = st.selectbox("Gender", options=["Male", "Female"])
                    gender = 1 if gender == "Male" else 0  # Encode gender as 1 for Male and 0 for Female
                    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120, step=1)
                    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200, step=1)
                    heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=80, step=1)

                    if st.button('Predict', key='dt_predict'):
                        dt_input_data = [[age, gender, blood_pressure, cholesterol, heart_rate]]

                        dt_prediction = rf_model.predict(dt_input_data)
                                
                        result = "Heart Disease Detected" if dt_prediction[0] == 1 else "No Heart Disease Detected"
                        st.markdown(f'The prediction result is: `{result}`')

                elif use_quantum_feature == "With Quantum Feature":
                    quantum_pattern_feature = st.number_input("Quantum Pattern Feature", min_value=0, max_value=100, value=50, step=1)
                    age = st.number_input("Age", min_value=0, max_value=100, value=50, step=1)
                    gender = st.selectbox("Gender", options=["Male", "Female"])
                    gender = 1 if gender == "Male" else 0  # Encode gender as 1 for Male and 0 for Female
                    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120, step=1)
                    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200, step=1)
                    heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=80, step=1)
                    
                    if st.button('Predict', key='quantum_predict'):
                        quantum_input_data = [[quantum_pattern_feature, age, gender, blood_pressure, cholesterol, heart_rate]]
                            
                        quantum_prediction = quantum.predict(quantum_input_data)
                            
                        result = "Heart Disease Detected" if quantum_prediction[0] == 1 else "No Heart Disease Detected"
                        st.markdown(f'The prediction result is: `{result}`')
        
    with col_pred[1]:
            st.subheader("Feature Similarity")
            st.markdown("### Predict Your Cluster")
            c_age = st.number_input("Age", min_value=0, max_value=100, value=50, step=1, key='cluster_age')
            c_blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120, step=1, key='cluster_bp')
            c_cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200, step=1, key='cluster_chol')
            c_heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=80, step=1, key='cluster_hr')

            numerical_features = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate']
            data_clustering = df[numerical_features].copy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_clustering)

            if st.button("Predict Cluster", key='predict_cluster'):
                user_input_data = [[c_age, c_blood_pressure, c_cholesterol, c_heart_rate]]

                scaled_user_input_data = scaler.transform(user_input_data)

                user_cluster = kmeans.predict(scaled_user_input_data)

                st.markdown(f"Your input data belongs to **Cluster {user_cluster[0]}**.")
                if user_cluster[0] == 0:
                    st.markdown("**Cluster 0**: Younger, lower-risk group.")
                elif user_cluster[0] == 1:
                    st.markdown("**Cluster 1**: Older, higher-risk group.")
                elif user_cluster[0] == 2:
                    st.markdown("**Cluster 2**: Middle-aged group with raised cholesterol and heart rate.")

###############
#conclusion page
if st.session_state.page_selection == "conclusion":
    st.title("Conclusion")
    st.markdown("CardioWise is a web application that helps in predicting the presence of heart disease in patients. The application uses machine learning models to predict the presence of heart disease in patients. The application also provides data visualization and exploratory data analysis to help users understand the data better.")
    st.markdown("This study effectively illustrates the value of machine learning in assessing cardiovascular risk.  **Supervised models** emphasize the relevance of the Quantum Pattern Feature in predicting heart disease, whereas age and gender are less predictive. **Unsupervised K-Means clustering** indicated patient groupings that needed more medical investigation for clinical relevance. The results revealed a **median patient age of 55-60**, different blood pressure and cholesterol distributions between genders, and a 40% heart disease prevalence.  The **Quantum Pattern Feature** has mild relationships with age and cholesterol, prompting additional inquiry. Overall, this study sheds light on prospective diagnostic, therapy, and prevention techniques for heart disease.")
    st.markdown("Note: This is just a prototype application and should not be used for medical diagnosis or treatment. Always consult a healthcare professional for medical advice and treatment.")