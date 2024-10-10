import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained NMF model and vectorizer
with open('nmf_model.pkl', 'rb') as model_file:
    nmf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Topic-to-category mapping
topics_to_categories = {0: "Technology", 1: "Politics", 2: "Entertainment", 3: "Sports", 4: "Business"}

# Streamlit App
st.title("News Text Classification")

# Input text from user
text_input = st.text_area("Enter the news text:")

if st.button("Classify"):
    if text_input:
        # Step 1: Transform the text using the TF-IDF vectorizer
        text_tfidf = vectorizer.transform([text_input])

        # Step 2: Get topic distribution from the NMF model
        topic_distribution = nmf_model.transform(text_tfidf)

        # Step 3: Get the most relevant topic (category) based on highest score
        predicted_topic = topic_distribution.argmax()
        predicted_category = topics_to_categories[predicted_topic]

        # Display the predicted category
        st.success(f"Predicted Category: {predicted_category}")
    else:
        st.error("Please enter some news text for classification.")
