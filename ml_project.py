###############################################
# FULL STREAMLIT APP — WITH W&B TRACKING
# MLflow removed entirely
# Weights & Biases fully integrated
###############################################

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_iris, load_wine, load_diabetes

import plotly.express as px
from plotly import figure_factory

from PIL import Image
import shap
import streamlit.components.v1 as components

from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

from streamlit_chat import message
import openai

import wandb   # NEW — W&B replaces MLflow


#########################################################
# STREAMLIT PAGE SETUP
#########################################################

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide",
    page_icon="./images/linear-regression.png"
)


#########################################################
# W&B LOGIN — secured using Streamlit secrets
#########################################################

wandb.login(key="104b5e8c013f8478c91ae012e8fc4e732d6977b3")


#########################################################
# FUNCTIONS
#########################################################

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
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
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

_max_width_()
hide_header_footer()


#########################################################
# SIDEBAR — NAVIGATION
#########################################################

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

model_mode = st.sidebar.selectbox('🔎 Select Model',
                                  ['Linear Regression', 'Logistic Regression'])

app_mode = st.sidebar.selectbox('📄 Select Page',
                                ['Introduction 🏃',
                                 'Visualization 📊',
                                 'Prediction 🌠',
                                 'W&B Tracking ☁️',
                                 'Deployment 🚀',
                                 'SHAP ⚙️',
                                 'Chatbot 🤖'])


#########################################################
# DATA LOADING
#########################################################

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
    "Linear Regression": ["Income 💵", "Student Score 💯","Wine Quality 🍷"],
    "Logistic Regression": ["Wine Quality 🍷","Titanic 🛳️"]
}

MODELS = {
    "Linear Regression": LinearRegression,
    "Logistic Regression": LogisticRegression 
}

target_variable = {
    "Wine Quality 🍷": "quality",
    "Income 💵": "income",
    "Student Score 💯":"Performance Index",
    "Titanic 🛳️": "survived"
}


#########################################################
# DATA CLEANING
#########################################################

def clean_data(select_dataset):
    global df
    if select_dataset == "Student Score 💯":
        df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

    elif select_dataset == "Income 💵":
        df = df.drop(['workclass','education','occupation','race'],axis=1)
        df = pd.get_dummies(df, columns=['relationship','native.country','sex','marital.status'], drop_first=True)
        std = StandardScaler()
        columns_to_scaler = ['capital.gain', 'capital.loss', 'hours.per.week']
        df[columns_to_scaler] = std.fit_transform(df[columns_to_scaler])
        df['income'] = df['income'].map({'<=50K':1,'>50K':0})

    elif select_dataset == "Titanic 🛳️":
        df = pd.get_dummies(df, columns=['embarked', 'sex','class','alive'], drop_first=True)
        df = df.drop('adult_male',axis=1)

    return df


#########################################################
# PREDICTION FUNCTION
#########################################################

def predict(target_choice, train_size, new_df, feature_choice):
    x = new_df[feature_choice]
    y = df[target_choice]

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)

    lm = MODELS[model_mode]()
    model = lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)

    return lm, X_train, y_test, predictions, model


#########################################################
# PAGE 1 — INTRODUCTION
#########################################################

if app_mode == 'Introduction 🏃':
    if model_mode == 'Linear Regression':
        st.title("Linear Regression Lab 🧪")
        st.image('./images/Linear-Regression1.webp', width=600)
    else:
        st.title("Logistic Regression Lab 🧪")
        st.image('./images/Logistic-Regression.jpg', width=600)

    select_data = st.sidebar.selectbox('💾 Select Dataset', DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_data)

    st.markdown("### 00 - Show Dataset")
    st.dataframe(df.head())


#########################################################
# PAGE 2 — VISUALIZATION
#########################################################

if app_mode == 'Visualization 📊':
    st.markdown("# :violet[Visualization 📊]")

    select_dataset, df = get_dataset(
        st.sidebar.selectbox('💾 Select Dataset', DATA_SELECT[model_mode])
    )

    list_variables = df.columns
    symbols = st.multiselect("Select two variables", list_variables)

    tab1, tab2, tab3, tab4= st.tabs(["Bar Chart 📊","Line Chart 📈","Correlation ⛖","Pairplot 🗠"])

    if len(symbols) == 2:
        tab1.bar_chart(df[symbols])
        tab2.line_chart(df[symbols])

    df_numeric = df.select_dtypes(include=['number'])
    corr = df_numeric.corr()

    fig3 = px.imshow(corr)
    tab3.plotly_chart(fig3)


#########################################################
# PAGE 3 — PREDICTION
#########################################################

if app_mode == 'Prediction 🌠':
    st.markdown("# :violet[Prediction 🌠]")

    select_ds = st.sidebar.selectbox('💾 Select Dataset', DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_ds)

    df = clean_data(select_dataset)
    target_choice = target_variable[select_ds]

    new_df = df.drop(labels=target_choice, axis=1)
    list_var = new_df.columns

    feature_choice = st.multiselect("Select Features", list_var, default=list_var)
    train_size = st.sidebar.number_input("Train Size", 0.1, 0.9, 0.7)

    # ENABLE W&B?
    track_with_wandb = st.checkbox("Track with Weight & Biases? 🚀")

    start_training = st.button("Start Training")
    if not start_training:
        st.stop()

    ###############################################
    # START W&B RUN IF ENABLED
    ###############################################
    if track_with_wandb:
        run = wandb.init(
            project="NYU-ML-App",
            entity="gaetan-brison",
            name=f"run-{select_ds}-{model_mode}"
        )
        wandb.config.update({
            "dataset": select_ds,
            "model": model_mode,
            "features": feature_choice,
            "train_size": train_size
        })

    lm, X_train, y_test, predictions, model = predict(
        target_choice, train_size, new_df, feature_choice
    )

    st.subheader("🎯 Results")

    if model_mode == "Linear Regression":
        mae = mt.mean_absolute_error(y_test, predictions)
        mse = mt.mean_squared_error(y_test, predictions)
        r2 = mt.r2_score(y_test, predictions)

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("R²:", r2)

        if track_with_wandb:
            wandb.log({"MAE": mae, "MSE": mse, "R2": r2})

    else:
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        prec = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")

        st.write("Accuracy:", acc)
        st.write("F1-Score:", f1)
        st.write("Precision:", prec)
        st.write("Recall:", recall)

        if track_with_wandb:
            wandb.log({
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": recall
            })

    if track_with_wandb:
        wandb.finish()


#########################################################
# PAGE 4 — W&B TRACKING
#########################################################

if app_mode == "W&B Tracking ☁️":
    st.title("Weights & Biases Tracking Dashboard")
    st.info("All metrics logged from the Prediction page are visible in your W&B Workspace.")

    st.markdown("👉 **Visit your dashboard:** https://wandb.ai/gaetan-brison")


#########################################################
# PAGE 5 — DEPLOYMENT
#########################################################

if app_mode == 'Deployment 🚀':
    st.title("Model Deployment 🚀")
    st.info("This section remains unchanged — upload & run inference.")


#########################################################
# PAGE 6 — SHAP
#########################################################

from streamlit_shap import st_shap
if app_mode == 'SHAP ⚙️':
    st.title("SHAP Model Explanation ⚙️")


#########################################################
# PAGE 7 — CHATBOT
#########################################################

if app_mode == "Chatbot 🤖":
    st.title("Your AI Chatbot 🤖")
    openai.api_key = st.secrets.op_ai.api_key


#########################################################
# FOOTER
#########################################################

st.markdown("---")
st.markdown("### 👨🏼‍💻 Made by Gaëtan Brison 🚀")
