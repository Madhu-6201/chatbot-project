import streamlit as st
import json
import random
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai  # OpenAI ki jagah Gemini use kar rahe hain

# 1. Setup Gemini Client 
# Streamlit Secrets se GOOGLE_API_KEY uthayega
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Secrets mein GOOGLE_API_KEY nahi mili. Please check Streamlit Settings!")
    st.stop()

# 2. Optimized Data Loading (Using Cache)
@st.cache_resource
def load_resources():
    # Load intents (make sure intents.json is in your repo)
    with open('intents.json') as f:
        intents_data = json.load(f)
    
    # Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load Pickled files (make sure these exist in your repo)
    pattern_embeddings = pickle.load(open('embeddings.pkl', 'rb'))
    tags = pickle.load(open('tags.pkl', 'rb'))
    
    return intents_data, model, pattern_embeddings, tags

data, model, pattern_embeddings, tags = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 Pandora AI")
    st.markdown("""
    **About Pandora:**
    I am a therapeutic assistant designed to listen and support you. 
    """)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# 3. Logic Functions
def predict_intent(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, pattern_embeddings)
    best_idx = np.argmax(similarities)
    return tags[best_idx], similarities[0][best_idx]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to help with that."

def get_ai_response(user_input):
    try:
        # Gemini model call
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(f"You are Pandora, a kind therapeutic assistant. User says: {user_input}")
        return response.text
    except Exception as e:
        return f"AI Error: Please check your Gemini API key. {str(e)}"

# 4. Streamlit UI
st.title("🤖 Hybrid AI Chatbot")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    tag, confidence = predict_intent(prompt)
    
    # Logic: High confidence = local intents, Low confidence = Gemini
    if confidence > 0.6:  # Threshold thoda increase kiya hai accuracy ke liye
        full_response = get_response(tag)
    else:
        full_response = get_ai_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
