import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Task 1 - Sentiment Analysis", layout="centered")

# 🎨 Custom Title
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:#FF6F61;'>📝 Task 1: The First Customer Review</h1>
        <h3 style='color:gray;'>Sentiment Analysis with Transformers 🤗</h3>
        <p style='font-size:16px; color:#555;'>Using <b>DistilBERT</b> from Hugging Face Hub for accurate results 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# Step 1: Load Model
st.subheader("⚡ Step 1: Load Pre-trained Model")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
st.success("✅ Model Loaded Successfully (DistilBERT)")

# Step 2: Take Input
st.subheader("📝 Step 2: Enter Customer Review")
user_input = st.text_area(" ", placeholder="Type a review, e.g., 'I love this product, it works perfectly!'")

# Step 3 & 4: Analyze Review
if st.button("✨ Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        sentiment = result["label"]
        score = round(result["score"] * 100, 2)

        st.markdown(f"**📌 Review Entered:** {user_input}")
        st.markdown(f"**📊 Sentiment Prediction:** `{sentiment}`")
        st.markdown(f"**🔎 Confidence Score:** {score}%")

        if sentiment == "POSITIVE":
            st.success("🌟 This is a **Positive Review!** 👍")
        else:
            st.error("⚠️ This is a **Negative Review!** 👎")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

st.write("---")

# Footer
st.markdown(
    """
    <div style='text-align: center; color:gray; font-size:14px;'>
        ✅ Task 1 Completed | Sentiment Analysis using Transformers 🤗 | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
