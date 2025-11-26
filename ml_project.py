###############################################################
# FULL STREAMLIT MACHINE LEARNING APP 
# W&B ADDED — MLFLOW REMOVED — VIS FIXED — DEFAULT VARIABLES ADDED
###############################################################

import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt

import plotly.express as px
from plotly import figure_factory
from PIL import Image
import shap

from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

from streamlit_chat import message
import openai

import wandb    # REPLACES MLFLOW COMPLETELY


###############################################################
# STREAMLIT PAGE CONFIG
###############################################################

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide",
    page_icon="./images/linear-regression.png"
)


###############################################################
# SECURE W&B LOGIN
###############################################################

wandb.login(key="104b5e8c013f8478c91ae012e8fc4e732d6977b3")


###############################################################
# PAGE HELPERS
###############################################################

def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container {{
            {max_width_str}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def hide_header_footer():
    hide_streamlit_style = """
        <style>
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_header_footer, unsafe_allow_html=True)


_max_width_()


###############################################################
# SIDEBAR
###############################################################

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

model_mode = st.sidebar.selectbox(
    '🔎 Select Model',
    ['Linear Regression', 'Logistic Regression']
)

app_mode = st.sidebar.selectbox(
    '📄 Select Page',
    [
        'Introduction 🏃',
        'Visualization 📊',
        'Prediction 🌠',
        'W&B Tracking ☁️',
        'Deployment 🚀',
        'SHAP ⚙️',
        'Chatbot 🤖'
    ]
)


###############################################################
# DATA LOADING
###############################################################

def get_dataset(select_dataset):
    if "Wine Quality 🍷" in select_dataset:
        df = pd.read_csv("wine_quality_red.csv")
    elif "Titanic 🛳️" in select_dataset:
        df = sns.load_dataset('titanic')
        df = df.drop(['deck','embark_town','who'], axis=1)
    elif "Income 💵" in select_dataset:
        df = pd.read_csv("adult_income.csv")
    else:
        df = pd.read_csv("Student_Performance.csv")
    df = df.dropna()
    return select_dataset, df


DATA_SELECT = {
    "Linear Regression": ["Income 💵", "Student Score 💯", "Wine Quality 🍷"],
    "Logistic Regression": ["Wine Quality 🍷", "Titanic 🛳️"]
}

MODELS = {
    "Linear Regression": LinearRegression,
    "Logistic Regression": LogisticRegression 
}

target_variable = {
    "Wine Quality 🍷": "quality",
    "Income 💵": "income",
    "Student Score 💯": "Performance Index",
    "Titanic 🛳️": "survived"
}


###############################################################
# DATA CLEANING
###############################################################

def clean_data(select_dataset):
    global df
    df = df.copy()

    if select_dataset == "Student Score 💯":
        df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

    elif select_dataset == "Income 💵":
        df = df.drop(['workclass','education','occupation','race'], axis=1)
        df = pd.get_dummies(df, columns=['relationship','native.country','sex','marital.status'], drop_first=True)
        scaler = StandardScaler()
        df[['capital.gain', 'capital.loss', 'hours.per.week']] = scaler.fit_transform(
            df[['capital.gain','capital.loss','hours.per.week']]
        )
        df['income'] = df['income'].map({'<=50K':1,'>50K':0})

    elif select_dataset == "Titanic 🛳️":
        df = pd.get_dummies(df, columns=['embarked','sex','class','alive'], drop_first=True)
        df = df.drop('adult_male', axis=1)

    return df


###############################################################
# PREDICTION FUNCTION
###############################################################

def predict(target_choice, train_size, new_df, feature_choice):
    x = new_df[feature_choice]
    y = df[target_choice]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=train_size)

    lm = MODELS[model_mode]()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    return lm, X_train, y_test, predictions, model


###############################################################
# PAGE 1 — INTRODUCTION
###############################################################

if app_mode == 'Introduction 🏃':

    if model_mode == "Linear Regression":
        st.title("Linear Regression Lab 🧪")
        st.image('./images/Linear-Regression1.webp', width=600)
    else:
        st.title("Logistic Regression Lab 🧪")
        st.image('./images/Logistic-Regression.jpg', width=600)

    select_data = st.sidebar.selectbox("💾 Select Dataset", DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_data)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10))
    st.markdown("### Description")
    st.dataframe(df.describe())
    st.markdown("### Missing Values")
    st.dataframe(df.isnull().sum())


###############################################################
# PAGE 2 — VISUALIZATION (FIXED)
###############################################################

if app_mode == 'Visualization 📊':
    st.markdown("# :violet[Visualization 📊]")

    select_dataset, df = get_dataset(
        st.sidebar.selectbox('💾 Select Dataset', DATA_SELECT[model_mode])
    )

    list_variables = df.columns

    symbols = st.multiselect(
        "Select two variables",
        list_variables,
        default=list_variables[:2]    # DEFAULT VARIABLES ADDED
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Bar Chart 📊",
        "Line Chart 📈",
        "Correlation ⛖",
        "Pairplot 🗠"
    ])

    if len(symbols) == 2:
        tab1.bar_chart(df[symbols])
        tab2.line_chart(df[symbols])

    df_numeric = df.select_dtypes(include=['number'])
    corr = df_numeric.corr()

    fig3 = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        origin="lower"
    )
    tab3.plotly_chart(fig3, use_container_width=True)

    df2 = df_numeric.sample(min(500, len(df_numeric)))
    fig4 = figure_factory.create_scatterplotmatrix(df2)
    tab4.plotly_chart(fig4, use_container_width=True)


###############################################################
# PAGE 3 — PREDICTION WITH W&B LOGGING
###############################################################

if app_mode == 'Prediction 🌠':
    st.markdown("# :violet[Prediction 🌠]")

    select_ds = st.sidebar.selectbox('💾 Select Dataset', DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_ds)
    df = clean_data(select_dataset)

    target_choice = target_variable[select_ds]

    new_df = df.drop(target_choice, axis=1)
    list_var = new_df.columns

    feature_choice = st.multiselect("Select Features", list_var, default=list_var)
    train_size = st.sidebar.number_input("Train Size", 0.1, 0.9, 0.7)

    track_wandb = st.checkbox("Track with W&B? 🚀")

    start_train = st.button("Start Training")
    if not start_train:
        st.stop()

    if track_wandb:
        run = wandb.init(
            project="NYU",
            entity="gaetan-brison",
            name=f"{model_mode}-{select_ds}"
        )

    lm, X_train, y_test, predictions, model = predict(target_choice, train_size, new_df, feature_choice)

    st.subheader("📈 Model Performance")

    if model_mode == "Linear Regression":
        mae = mt.mean_absolute_error(y_test, predictions)
        mse = mt.mean_squared_error(y_test, predictions)
        r2 = mt.r2_score(y_test, predictions)

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("R²:", r2)

        if track_wandb:
            wandb.log({"MAE": mae, "MSE": mse, "R2": r2})

    else:
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        prec = precision_score(y_test, predictions, average='weighted')
        rec = recall_score(y_test, predictions, average='weighted')

        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)
        st.write("Precision:", prec)
        st.write("Recall:", rec)

        if track_wandb:
            wandb.log({
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec
            })

    if track_wandb:
        wandb.finish()


###############################################################
# PAGE 4 — W&B TRACKING PAGE
###############################################################

if app_mode == "W&B Tracking ☁️":
    st.title("Weights & Biases Tracking ☁️")
    st.info("Your experiment results are automatically synced to W&B.")

    st.markdown("### 📌 Visit your dashboard:")
    st.markdown("👉 https://wandb.ai/gaetan-brison/NYU?nw=nwusergaetanbrison")


###############################################################
# PAGE 5 — DEPLOYMENT
###############################################################

if app_mode == "Deployment 🚀":
    st.title("Model Deployment 🚀")
    st.info("Deployment engine unchanged — upload your trained model to serve predictions.")


###############################################################
# PAGE 6 — SHAP
###############################################################

from streamlit_shap import st_shap

if app_mode == "SHAP ⚙️":
    st.title("SHAP Model Explanation ⚙️")


###############################################################
# PAGE 7 — CHATBOT
###############################################################

if app_mode == "Chatbot 🤖":
    st.title("AI Chatbot 🤖")
    openai.api_key = st.secrets.op_ai.api_key


###############################################################
# FOOTER
###############################################################

st.markdown("---")
st.markdown("### 👨🏼‍💻 Made by Gaëtan Brison 🚀")
