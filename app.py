import nltk
nltk.download('punkt_tab')

# Download the required resources for the lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt') # Common to need this for tokenization as well
import streamlit as st
import pickle
import json
import random
import nltk
import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load files
model = pickle.load(open('model.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

encoder = pickle.load(open('label_encoder.pkl', 'rb'))

with open('intents.json') as file:
    data = json.load(file)

# Clean text
def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    words = nltk.word_tokenize(text)
    
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalnum()
    ]
    
    return " ".join(words)

# Predict
def predict_intent(text):
    
    cleaned = clean_text(text)
    
    vector = vectorizer.transform([cleaned]).toarray()
    
    prediction = model.predict(vector)[0]
    
    probabilities = model.predict_proba(vector)[0]
    
    confidence = max(probabilities)
    
    tag = encoder.inverse_transform([prediction])[0]
    
    return tag, confidence

# Response
def get_response(tag):
    
    for intent in data['intents']:
        
        if intent['tag'] == tag:
            
            return random.choice(intent['responses'])

# Streamlit UI
st.title("🤖 AI Chatbot")

user_input = st.text_input("You:")

if user_input:
    
    tag, confidence = predict_intent(user_input)
    
    if confidence > 0.4:
        
        response = get_response(tag)
    
    else:
        
        response = "Sorry 😅 I didn't understand."
    
    st.write("Bot:", response)
