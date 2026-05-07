import streamlit as st
import pickle
import json
import random
import nltk
import re
from nltk.stem import WordNetLemmatizer

# --- NLTK data download (Zaroori files) ---
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

load_nltk()
lemmatizer = WordNetLemmatizer()

# --- 1. Files Load Karein ---
# Make sure ye saari files aapke folder mein मौजूद ho
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError:
    st.error("Error: Model files ya intents.json nahi mili! Pehle model train karein.")

# --- 2. AI Logic Functions ---
def clean_text(text):
    text = text.lower()
    # Repeating characters ko handle karna (e.g., "Hiiiii" -> "Hii")
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
    return "I'm sorry, I don't have a response for that yet."

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 My Smart AI Assistant")
st.markdown("---")

# Chat history initialize karein
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani chat screen par dikhayein
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input aur Response Logic ---
if prompt := st.chat_input("Mujhse kuch puchiye..."):
    # User message display aur save
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Model se prediction lena
    tag, confidence = predict_intent(prompt)
    
    # --- CONFIDENCE LOGIC (Main Fix) ---
    # Agar confidence 0.1 se upar hai toh reply do (taaki Hii/Hello miss na ho)
    if confidence > 0.1: 
        response = get_response(tag)
    else:
        response = "Mafi chahta hoon, mujhe ye samajh nahi aaya. 😅 Kya aap dusre shabdon mein bol sakte hain?"

    # Bot response display aur save
    with st.chat_message("assistant"):
        st.markdown(response)
        # Debugging ke liye (optional): st.caption(f"Confidence: {confidence:.2f}")
        
    st.session_state.messages.append({"role": "assistant", "content": response})