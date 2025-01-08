import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import os
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
uploaded_file = 'ev_charging_dataset.csv'

if not os.path.exists(uploaded_file):
    st.error("Dataset not found! Please upload the dataset.")
    st.stop()

data = pd.read_csv(uploaded_file)

# Remove problematic columns
date_time_columns = ['Date', 'Time', 'DateTime']  # Adjust column names as per your dataset
for col in date_time_columns:
    if col in data.columns:
        data = data.drop(columns=[col])

# Ensure numeric data only
data = data.select_dtypes(include=['float64', 'int64'])

# Streamlit App
st.title("EV Charging Load Prediction")

st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose an operation:", [
    "Display Data Info",
    "Data Cleaning",
    "EDA",
    "Encoding",
    "Train and Predict"
])

if option == "Display Data Info":
    st.header("Dataset Information")
    st.write("### Data Preview")
    st.dataframe(data.head())
    st.write("### Dataset Info")

    buffer = StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("### Summary Statistics")
    st.write(data.describe())

elif option == "Data Cleaning":
    st.header("Data Cleaning")
    st.write("### Missing Values")
    st.write(data.isnull().sum())

    clean_option = st.radio("Handle Missing Values:",
                            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
    if st.button("Clean Data"):
        if clean_option == "Drop rows":
            data = data.dropna()
        elif clean_option == "Fill with mean":
            data = data.fillna(data.mean())
        elif clean_option == "Fill with median":
            data = data.fillna(data.median())
        elif clean_option == "Fill with mode":
            data = data.fillna(data.mode().iloc[0])
        st.success("Missing values handled successfully!")
        st.write(data.isnull().sum())

elif option == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("### Column Analysis")
    column_to_analyze = st.selectbox("Select a column for analysis:", data.columns)

    if column_to_analyze:
        st.write(f"### Data Distribution for {column_to_analyze}")
        st.bar_chart(data[column_to_analyze])

        st.write(f"### Box Plot for {column_to_analyze}")
        fig, ax = plt.subplots()
        sns.boxplot(y=data[column_to_analyze], ax=ax)
        st.pyplot(fig)

        st.write(f"### Histogram for {column_to_analyze}")
        fig, ax = plt.subplots()
        sns.histplot(data[column_to_analyze], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Relationship Between Columns")
    col1 = st.selectbox("Select first column:", data.columns, key="eda_col1")
    col2 = st.selectbox("Select second column:", data.columns, key="eda_col2")

    if st.button("Show Visual"):
        st.write(f"### Relationship between {col1} and {col2}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[col1], y=data[col2], ax=ax)
        ax.set_title(f"Scatter Plot: {col1} vs {col2}")
        st.pyplot(fig)

elif option == "Encoding":
    st.header("Encoding Text Data")
    st.write("All columns are numeric. No encoding needed.")

elif option == "Train and Predict":
    st.header("Train and Predict")
    target = st.selectbox("Select target column (Y):", data.columns)
    features = st.multiselect("Select feature columns (X):", [col for col in data.columns if col != target])

    if features and target:
        X = data[features]
        y = data[target]

        model_choice = st.selectbox("Select a model:", ["Linear Regression", "Random Forest", "SVR"])

        test_size_options = {
            "70:30": 0.3,
            "80:20": 0.2,
            "90:10": 0.1
        }
        test_size_ratio = st.radio("Select test size ratio:", list(test_size_options.keys()))
        test_size = test_size_options[test_size_ratio]

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()
            elif model_choice == "SVR":
                model = SVR()

            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.session_state['features'] = features
            st.success(f"{model_choice} trained successfully!")

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            st.write("### Model Performance")
            st.write(f"Train Score: {train_score:.2f}")
            st.write(f"Test Score: {test_score:.2f}")

        if "model" in st.session_state and "features" in st.session_state:
            st.write("### Make Predictions")
            user_input = {}
            for feature in st.session_state['features']:
                value = st.number_input(f"Enter value for {feature}", key=f"input_{feature}")
                user_input[feature] = value

            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                prediction = st.session_state['model'].predict(input_df)[0]
                st.write(f"Predicted Charging Load (kW): {prediction}")
