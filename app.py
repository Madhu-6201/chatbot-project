import streamlit as st
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer

# -------------------- Setup --------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

lemmatizer = WordNetLemmatizer()

# Download NLTK data (important for deployment)
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

load_nltk()

# -------------------- Load Files --------------------
@st.cache_resource
def load_files():
    model = pickle.load(open('model.pkl', 'rb'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    
    with open('intents.json') as file:
        data = json.load(file)
    
    return model, words, classes, data

model, words, classes, data = load_files()

# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = text.lower().strip()
    
    replacements = {
        "hii": "hi",
        "heyy": "hey",
        "hlo": "hello",
        "hy": "hi"
    }
    
    return replacements.get(text, text)

# -------------------- NLP Functions --------------------
def bag_of_words(sentence):
    sentence = clean_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    probs = model.predict_proba([bow])[0]
    
    max_prob = max(probs)
    
    # Confidence threshold
    if max_prob < 0.3:
        return "unknown"
    
    return classes[np.argmax(probs)]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# -------------------- UI --------------------
st.title("🤖 AI Chatbot")
st.write("Type your message below:")

user_input = st.text_input("You:")

if user_input:
    try:
        tag = predict_class(user_input)
        
        # Debug line (you can remove later)
        st.write("Predicted tag:", tag)

        if tag == "unknown":
            st.write("Bot: Sorry, I didn't understand 😅")
        else:
            response = get_response(tag)
            st.write("Bot:", response)

    except Exception as e:
        st.error(f"Error: {e}")