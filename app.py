import streamlit as st
import json
import random
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# 🔑 Add your API key here
client = OpenAI(api_key="YOUR_API_KEY")

# Load data
with open('intents.json') as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

pattern_embeddings = pickle.load(open('embeddings.pkl', 'rb'))
tags = pickle.load(open('tags.pkl', 'rb'))

# Predict
def predict_intent(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, pattern_embeddings)
    
    best_idx = np.argmax(similarities)
    return tags[best_idx], similarities[0][best_idx]

# Dataset response
def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# 🔥 AI fallback
def get_ai_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# UI
st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖")

st.title("🤖 Hybrid AI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if user_input:
    
    tag, confidence = predict_intent(user_input)
    
    if confidence > 0.5:
        response = get_response(tag)
    else:
        response = get_ai_response(user_input)  # 🔥 fallback
    
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

for sender, msg in st.session_state.history:
    st.write(f"{sender}: {msg}")