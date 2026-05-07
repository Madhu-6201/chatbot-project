import streamlit as st
import pickle
import json
import random
import nltk
import re
from nltk.stem import WordNetLemmatizer

# NLTK data download
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# --- Files Load Karein ---
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))
with open('intents.json') as file:
    data = json.load(file)

# --- AI Logic Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'(.)\1+', r'\1\1', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    return " ".join(words)

def predict_intent(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = max(probabilities)
    tag = encoder.inverse_transform([prediction])[0]
    return tag, confidence

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand."

# --- Streamlit UI aur History Logic ---
st.title("🤖 AI Chatbot")

# 1. History initialize karein
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Purani chat screen par dikhayein
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Naya Input handle karein
if prompt := st.chat_input("Yahan kuch likhein..."):
    # User message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Prediction
    tag, confidence = predict_intent(prompt)
    if confidence > 0.4:
        response = get_response(tag)
    else:
        response = "Sorry 😅 I didn't understand."

    # Bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
