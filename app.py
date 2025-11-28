# app.py

# app.py

# app.py
import streamlit as st
import pickle
import re
import nltk
import feedparser
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# Clear input function
def clear_text():
    st.session_state.user_input = ""

# Load sample data
@st.cache_data
def load_samples():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['label'] = 'Fake'
    true['label'] = 'Real'
    combined = pd.concat([fake[['text', 'label']], true[['text', 'label']]])
    return combined.sample(10).reset_index(drop=True)

# Streamlit layout
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")

option = st.radio("Choose View", ["Fake News Detection", "Daily Indian News Feed (The Hindu)"])

if option == "Fake News Detection":
    st.write("Enter a news article or headline below to determine if it's fake or real.")

    # Toggle to show sample data
    if st.checkbox("Show Sample News Data (Real & Fake)"):
        st.subheader("📄 Sample News Data")
        st.dataframe(load_samples())

    st.subheader("✍️ Enter Custom News Text")
    user_input = st.text_area("Enter News Text", height=200, key="user_input")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please enter some news text.")
            else:
                cleaned = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                st.success("🟢 Real News" if prediction == 1 else "🔴 Fake News")
    with col2:
        st.button("Clear", on_click=clear_text)

elif option == "Daily Indian News Feed (The Hindu)":
    st.subheader("🗞️ Daily Indian News Feed (The Hindu)")

    rss_url = "https://www.thehindu.com/news/national/feeder/default.rss"
    feed = feedparser.parse(rss_url)

    if feed.entries:
        num_cols = 3
        entries_to_show = feed.entries[:10]
        for i in range(0, len(entries_to_show), num_cols):
            cols = st.columns(num_cols)
            for j, entry in enumerate(entries_to_show[i:i + num_cols]):
                with cols[j]:
                    st.markdown(f"**📰 {entry.title}**")
                    st.write(entry.summary if 'summary' in entry else '')
                    st.markdown(f"[Read more]({entry.link})")
                    st.markdown("---")
    else:
        st.info("No news found in the RSS feed.")
