###############################################################
# FULL STREAMLIT MACHINE LEARNING APP 
# W&B BUTTON • NO DEPLOYMENT • XGBOOST & RF • SHAP FIXED
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import xgboost as xgb

import plotly.express as px
from plotly import figure_factory
from PIL import Image
import shap
from streamlit_shap import st_shap

from streamlit_chat import message
import openai
import wandb


###############################################################
# STREAMLIT PAGE CONFIG
###############################################################

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide",
    page_icon="./images/linear-regression.png"
)


###############################################################
# W&B LOGIN
###############################################################

wandb.login(key="104b5e8c013f8478c91ae012e8fc4e732d6977b3")


###############################################################
# UI HELPERS
###############################################################

def _max_width_():
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            max-width: 1100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_max_width_()


###############################################################
# SIDEBAR
###############################################################

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

model_mode = st.sidebar.selectbox(
    '🔎 Select Model',
    [
        'Linear Regression',
        'Logistic Regression',
        'Random Forest',
        'XGBoost'
    ]
)

app_mode = st.sidebar.selectbox(
    '📄 Select Page',
    [
        'Introduction 🏃',
        'Visualization 📊',
        'Prediction 🌠',
        'W&B Tracking ☁️',
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
    "Logistic Regression": ["Wine Quality 🍷", "Titanic 🛳️"],
    "Random Forest": ["Wine Quality 🍷", "Titanic 🛳️", "Income 💵"],
    "XGBoost": ["Wine Quality 🍷", "Titanic 🛳️", "Income 💵"]
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
        df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(
            lambda x: 1 if x == 'Yes' else 0)

    elif select_dataset == "Income 💵":
        df = df.drop(['workclass','education','occupation','race'], axis=1)
        df = pd.get_dummies(df,
                            columns=['relationship','native.country','sex','marital.status'],
                            drop_first=True)
        scaler = StandardScaler()
        df[['capital.gain','capital.loss','hours.per.week']] = scaler.fit_transform(
            df[['capital.gain','capital.loss','hours.per.week']]
        )
        df['income'] = df['income'].map({'<=50K':1,'>50K':0})

    elif select_dataset == "Titanic 🛳️":
        df = pd.get_dummies(df, columns=['embarked','sex','class','alive'], drop_first=True)
        df = df.drop('adult_male', axis=1)

    return df


###############################################################
# MODEL TRAINING FUNCTION
###############################################################

def train_model(model_mode, X_train, y_train):
    if model_mode == "Linear Regression":
        model = LinearRegression()

    elif model_mode == "Logistic Regression":
        model = LogisticRegression(max_iter=2000)

    elif model_mode == "Random Forest":
        model = RandomForestClassifier(n_estimators=300) if y_train.nunique() <= 2 \
            else RandomForestRegressor(n_estimators=300)

    elif model_mode == "XGBoost":
        model = xgb.XGBClassifier(eval_metric='logloss') if y_train.nunique() <= 2 \
            else xgb.XGBRegressor()

    model.fit(X_train, y_train)
    return model


###############################################################
# PREDICTION FUNCTION
###############################################################

def predict(model, X_test):
    return model.predict(X_test)


###############################################################
# 1 — INTRODUCTION
###############################################################

if app_mode == 'Introduction 🏃':
    st.title(model_mode + " Lab")

    select_data = st.sidebar.selectbox("💾 Select Dataset", DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_data)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Description")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())


###############################################################
# 2 — VISUALIZATION
###############################################################

if app_mode == 'Visualization 📊':
    st.title("Visualization 📊")

    select_dataset, df = get_dataset(
        st.sidebar.selectbox("💾 Select Dataset", DATA_SELECT[model_mode])
    )

    list_vars = df.columns

    symbols = st.multiselect(
        "Select two variables",
        list_vars,
        default=list_vars[:2]
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Bar Chart", "Line Chart", "Correlation", "Pairplot"]
    )

    if len(symbols) == 2:
        tab1.bar_chart(df[symbols])
        tab2.line_chart(df[symbols])

    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr()

    fig_corr = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        origin="lower"
    )
    tab3.plotly_chart(fig_corr)

    fig_pair = figure_factory.create_scatterplotmatrix(num_df.sample(min(500, len(num_df))))
    tab4.plotly_chart(fig_pair)


###############################################################
# 3 — PREDICTION + W&B
###############################################################

if app_mode == 'Prediction 🌠':
    st.title("Prediction 🌠")

    select_ds = st.sidebar.selectbox("💾 Select Dataset", DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_ds)
    df = clean_data(select_dataset)

    target = target_variable[select_ds]
    X = df.drop(columns=[target])
    y = df[target]

    features = st.multiselect("Select Features", X.columns, default=X.columns)
    X = X[features]

    test_size = st.sidebar.number_input("Train Size", 0.1, 0.9, 0.7)

    track_wandb = st.checkbox("Track experiment with W&B? 🚀")

    start = st.button("Train Model")
    if not start:
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if track_wandb:
        run = wandb.init(
            project="NYU",
            entity="gaetan-brison",
            name=f"{model_mode}-{select_ds}",
            config={
                "model": model_mode,
                "dataset": select_ds,
                "features": list(features)
            }
        )

    model = train_model(model_mode, X_train, y_train)
    preds = predict(model, X_test)

    st.subheader("Results")

    if y.nunique() > 2:
        mae = mt.mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write("MAE:", mae)
        st.write("R²:", r2)
        if track_wandb:
            wandb.log({"MAE": mae, "R2": r2})
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)
        if track_wandb:
            wandb.log({"Accuracy": acc, "F1": f1})

    if track_wandb:
        wandb.finish()


###############################################################
# 4 — W&B Dashboard Button
###############################################################

if app_mode == "W&B Tracking ☁️":
    st.title("🏋️ Weights & Biases Experiment Tracking")

    st.info("Click the button below to view your dashboard:")

    st.link_button(
        "🔗 Open W&B Dashboard",
        "https://wandb.ai/gaetan-brison/NYU?nw=nwusergaetanbrison"
    )


###############################################################
# 5 — SHAP PAGE
###############################################################

if app_mode == "SHAP ⚙️":
    st.title("SHAP Model Explainability ⚙️")

    st.warning("Run a Prediction first so the model & data load here.")

    select_ds = st.sidebar.selectbox("Dataset", DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_ds)
    df = clean_data(select_dataset)

    target = target_variable[select_ds]
    X = df.drop(columns=[target])
    y = df[target]

    model = train_model(model_mode, X, y)

    try:
        explainer = shap.Explainer(model, X.sample(100))
        shap_values = explainer(X.sample(100))

        st.subheader("SHAP Summary Plot")
        st_shap(shap.plots.beeswarm(shap_values), height=600)

        st.subheader("First Prediction Waterfall")
        st_shap(shap.plots.waterfall(shap_values[0]), height=600)

    except Exception as e:
        st.error(f"SHAP failed: {e}")


###############################################################
# CHATBOT
###############################################################

if app_mode == "Chatbot 🤖":
    st.title("AI Chatbot 🤖")
    openai.api_key = st.secrets.op_ai.api_key


###############################################################
# FOOTER
###############################################################

st.markdown("---")
st.markdown("### 👨🏼‍💻 Made by Gaëtan Brison 🚀")
