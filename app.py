import streamlit as st
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load files
model = pickle.load(open('model.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json') as file:
    data = json.load(file)

# Functions
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    probs = model.predict_proba([bow])[0]
    
    max_prob = max(probs)
    if max_prob < 0.5:
        return "unknown"
    
    return classes[np.argmax(probs)]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# UI
st.title("🤖 AI Chatbot")
st.write("Type your message below:")

user_input = st.text_input("You:")

if user_input:
    tag = predict_class(user_input)
    
    if tag == "unknown":
        st.write("Bot: Sorry, I didn't understand 😅")
    else:
        response = get_response(tag)
        st.write("Bot:", response)