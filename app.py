# netflix_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Netflix Movie Ratings Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Helper: Load Netflix Dataset (SAFE)
# -------------------
@st.cache_data(show_spinner=False)
def load_netflix_data(file, limit=1000):
    data = []

    if file.name.endswith(".zip"):
        with zipfile.ZipFile(file) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                current_movie = None
                for raw_line in f:
                    if len(data) >= limit:
                        break

                    line = raw_line.decode("latin1").strip()

                    if line.endswith(":"):
                        current_movie = int(line.replace(":", ""))
                    else:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                user_id = int(parts[0])
                                rating = int(parts[1])
                                data.append([user_id, rating, current_movie])
                            except:
                                continue

        return pd.DataFrame(
            data, columns=["User_ID", "Rating", "Movie_ID"]
        )

    else:
        df = pd.read_csv(file, encoding="latin1")
        return df.head(limit)

# -------------------
# Sidebar Navigation
# -------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Upload Dataset", "Dataset Overview", "EDA", "Modeling"]
)

# -------------------
# Upload Dataset Page
# -------------------
if page == "Upload Dataset":
    st.title("📂 Upload your Netflix Ratings Dataset")

    rows_limit = st.slider(
        "Rows to load (for fast performance)",
        100, 5000, 1000, step=100
    )

    uploaded_file = st.file_uploader(
        "Upload Netflix ZIP or CSV file",
        type=["zip", "csv"]
    )

    if uploaded_file is not None:
        with st.spinner("Loading dataset safely..."):
            df = load_netflix_data(uploaded_file, rows_limit)
            st.session_state["df"] = df

        st.success("✅ Dataset loaded successfully!")
        st.write("Preview:")
        st.dataframe(df.head(20))

    else:
        st.info("Please upload a dataset to continue")

# -------------------
# Dataset Check
# -------------------
if "df" not in st.session_state:
    if page != "Upload Dataset":
        st.warning("⚠️ Upload dataset first")
        st.stop()
else:
    df = st.session_state["df"]

# -------------------
# Dataset Overview
# -------------------
if page == "Dataset Overview":
    st.title("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", len(df))
    col2.metric("Users", df["User_ID"].nunique())
    col3.metric("Movies", df["Movie_ID"].nunique())
    col4.metric("Avg Rating", round(df["Rating"].mean(), 2))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# -------------------
# EDA Section
# -------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("⭐ Rating Distribution")
    fig, ax = plt.subplots()
    df["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("👤 Ratings per User")
    fig, ax = plt.subplots()
    df.groupby("User_ID")["Rating"].count().hist(bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("🎥 Ratings per Movie")
    fig, ax = plt.subplots()
    df.groupby("Movie_ID")["Rating"].count().hist(bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Boxplot of Ratings")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Rating"], ax=ax)
    st.pyplot(fig)

elif page == "Modeling":
    st.title("🤖 Modeling & Evaluation")

    X = df[["User_ID", "Movie_ID"]]
    y = df["Rating"]

    # -------------------
    # Handle missing values
    # -------------------
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y.values, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox(
        "Select Model",
        ["Linear Regression", "Decision Tree"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        max_depth = st.slider("Max Depth", 2, 20, 10)
        min_samples_split = st.slider("Min Samples Split", 2, 50, 10)
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

    # -------------------
    # Train Model
    # -------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------
    # Evaluation Metrics
    # -------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📏 Evaluation Metrics")
    st.metric("MAE", round(mae, 3))
    st.metric("RMSE", round(rmse, 3))
    st.metric("R² Score", round(r2, 4))

    st.subheader("🔍 Predictions vs Actual (First 20 Records)")
    st.dataframe(
        pd.DataFrame({
            "Actual": y_test[:20],
            "Predicted": y_pred[:20]
        })
    )

