import streamlit as st
import json
import random
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 1. Setup OpenAI Client 
# Tip: Use st.secrets for production instead of hardcoding keys
client = OpenAI(api_key="YOUR_API_KEY")

# 2. Optimized Data Loading (Using Cache)
@st.cache_resource
def load_resources():
    # Load intents
    with open('intents.json') as f:
        intents_data = json.load(f)
    
    # Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load Pickled files
    pattern_embeddings = pickle.load(open('embeddings.pkl', 'rb'))
    tags = pickle.load(open('tags.pkl', 'rb'))
    
    return intents_data, model, pattern_embeddings, tags

# Initialize resources
data, model, pattern_embeddings, tags = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 Pandora AI")
    st.markdown("""
    **About Pandora:**
    I am a therapeutic assistant designed to listen and support you. 
    
    *Note: I am an AI, not a replacement for professional mental healthcare.*
    """)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.intro_given = False
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: Please check your OpenAI API key. {str(e)}"

# 4. Streamlit UI
st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖")
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
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    tag, confidence = predict_intent(prompt)
    
    # Logic: If confidence is high, use local intents. Else, use OpenAI.
    if confidence > 0.5:
        full_response = get_response(tag)
    else:
        full_response = get_ai_response(prompt)

    # Add bot response to history
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
