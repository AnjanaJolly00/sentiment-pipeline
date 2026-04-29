import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Review Intelligence Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload a CSV file
2. Select the column with review text
3. Click 'Analyze Sentiment'
""")

st.sidebar.write("### Sample format")
st.sidebar.code("""
review
This product is amazing
Worst purchase ever
""")

# Title
st.title("Amazon Review Sentiment Intelligence Dashboard")

st.write("Analyze customer sentiment using a fine-tuned DistilBERT model.")

# Manual testing section (NEW)
st.subheader("Try Single Review")

user_input = st.text_area("Enter a review")

if st.button("Predict Sentiment"):
    if user_input:
        response = requests.post(
            f"{API_URL}/predict",
            json={"review": user_input}
        )
        result = response.json()

        st.success(f"Sentiment: {result['sentiment']}")
        st.write(f"Confidence: {result['confidence']}")
    else:
        st.warning("Please enter a review")

st.divider()

# Upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    text_column = st.selectbox("Select review text column", df.columns)

    if st.button("Analyze Sentiment"):
        reviews = df[text_column].dropna().astype(str).tolist()

        response = requests.post(
            f"{API_URL}/predict-batch",
            json={"reviews": reviews}
        )

        results = response.json()["results"]

        df_result = df.dropna(subset=[text_column]).copy()
        df_result["sentiment"] = [r["sentiment"] for r in results]
        df_result["confidence"] = [r["confidence"] for r in results]

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Positive Reviews", (df_result["sentiment"] == "Positive").sum())
        col2.metric("Negative Reviews", (df_result["sentiment"] == "Negative").sum())

        # Charts
        st.subheader("Sentiment Distribution")
        fig = px.histogram(df_result, x="sentiment", color="sentiment")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Distribution")
        fig2 = px.histogram(df_result, x="confidence", nbins=20)
        st.plotly_chart(fig2, use_container_width=True)

        # Tables
        st.subheader("Top Positive Reviews")
        st.dataframe(
            df_result[df_result["sentiment"] == "Positive"]
            .sort_values("confidence", ascending=False)
            .head(5)
        )

        st.subheader("Top Negative Reviews")
        st.dataframe(
            df_result[df_result["sentiment"] == "Negative"]
            .sort_values("confidence", ascending=False)
            .head(5)
        )