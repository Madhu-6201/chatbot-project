import streamlit as st
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# -------------------- NLTK SETUP --------------------
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

load_nltk()

lemmatizer = WordNetLemmatizer()

# -------------------- LOAD FILES --------------------
@st.cache_resource
def load_files():
    model = pickle.load(open('model.pkl', 'rb'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    
    with open('intents.json') as file:
        data = json.load(file)
    
    return model, words, classes, data

model, words, classes, data = load_files()

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower().strip()
    
    replacements = {
        "hii": "hi",
        "heyy": "hey",
        "hlo": "hello",
        "hy": "hi"
    }
    
    return replacements.get(text, text)

# -------------------- NLP FUNCTIONS --------------------
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
    
    if max_prob < 0.3:
        return "unknown"
    
    return classes[np.argmax(probs)]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# -------------------- UI DESIGN --------------------
st.title("🤖 AI Chatbot")
st.caption("Chat like WhatsApp 💬")

# Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot response
    try:
        tag = predict_class(user_input)

        # Debug (optional - remove later)
        # st.write("Predicted:", tag)

        if tag == "unknown":
            bot_response = "Sorry, I didn't understand 😅"
        else:
            bot_response = get_response(tag)

    except Exception as e:
        bot_response = f"Error: {e}"

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})