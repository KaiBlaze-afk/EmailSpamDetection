import streamlit as st
import pickle
import os

MODEL_PATH = "spam_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        vectorizer, model = pickle.load(f)
else:
    st.error("Model file not found! Please ensure 'spam_model.pkl' is in the same directory.")
    st.stop()

st.title("Spam Message Classifier")
st.write("Enter a message below to check if it's spam or not.")

user_input = st.text_area("Enter message:")

if st.button("Classify"):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        
        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT spam.")
    else:
        st.warning("Please enter a message to classify.")
