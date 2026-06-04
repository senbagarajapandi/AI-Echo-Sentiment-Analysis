
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATASET
# ---------------------------

df = pd.read_csv("/content/chatgpt_style_reviews_dataset.csv")

# ---------------------------
# LOAD MODEL
# ---------------------------

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------
# TITLE
# ---------------------------

st.title("AI Echo: Your Smartest Conversational Partner")

st.write("Sentiment Analysis of ChatGPT User Reviews")

# ---------------------------
# DATASET PREVIEW
# ---------------------------

st.header("Dataset Preview")

st.dataframe(df)

# ---------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------

st.header("Sentiment Distribution")

sentiment_counts = df["SENTIMENT"].value_counts()

fig, ax = plt.subplots()

ax.bar(
    sentiment_counts.index,
    sentiment_counts.values
)

ax.set_title("Sentiment Distribution")

ax.set_xlabel("Sentiment")

ax.set_ylabel("Count")

st.pyplot(fig)

# ---------------------------
# RATING DISTRIBUTION
# ---------------------------

st.header("Rating Distribution")

rating_counts = df["RATING"].value_counts().sort_index()

fig, ax = plt.subplots()

ax.bar(
    rating_counts.index,
    rating_counts.values
)

ax.set_title("Rating Distribution")

ax.set_xlabel("Rating")

ax.set_ylabel("Count")

st.pyplot(fig)

# ---------------------------
# PLATFORM ANALYSIS
# ---------------------------

st.header("Platform Analysis")

platform_rating = (
    df.groupby("PLATFORM")["RATING"]
    .mean()
)

fig, ax = plt.subplots()

platform_rating.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Rating by Platform")

st.pyplot(fig)

# ---------------------------
# VERSION ANALYSIS
# ---------------------------

st.header("Version Analysis")

version_rating = (
    df.groupby("VERSION")["RATING"]
    .mean()
)

fig, ax = plt.subplots()

version_rating.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Rating by Version")

st.pyplot(fig)

# ---------------------------
# SENTIMENT PREDICTION
# ---------------------------

st.header("Predict Sentiment")

user_review = st.text_area(
    "Enter Review"
)

if st.button("Predict Sentiment"):

    review_vector = tfidf.transform(
        [user_review]
    )

    prediction = model.predict(
        review_vector
    )

    st.success(
        f"Predicted Sentiment: {prediction[0]}"
    )

# ---------------------------
# MODEL PERFORMANCE
# ---------------------------

st.header("Model Performance")

st.write("Model Used: Logistic Regression")

st.write("Evaluation Metrics Generated During Training")

# ---------------------------
# PROJECT INSIGHTS
# ---------------------------

st.header("Insights")

st.write(
    """
    • Analyze customer satisfaction

    • Identify positive and negative reviews

    • Compare ratings across platforms

    • Compare ratings across versions

    • Predict sentiment from user reviews
    """
)
