import streamlit as st
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pandora AI", page_icon="🤖", layout="centered")

# --- INITIALIZING SESSION STATE (MEMORY) ---
# This keeps the chat history alive even when the app refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "intro_given" not in st.session_state:
    st.session_state.intro_given = False

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

# --- MAIN INTERFACE ---
st.title("Pandora: Your Personal AI")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & LOGIC ---
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Logic to generate a smart response
    user_text = prompt.lower()
    
    # Logic: Capture Name
    if "my name is" in user_text:
        name = prompt.split("is")[-1].strip().capitalize()
        st.session_state.user_name = name
        response = f"It's lovely to meet you, {name}! How has your day been so far?"
    
    # Logic: Avoid Repetitive Introductions
    elif any(word in user_text for word in ["who are you", "your name", "what are you"]):
        if st.session_state.intro_given:
            response = "As I mentioned, I'm Pandora. I'm here to chat and support you however I can!"
        else:
            response = "I'm Pandora, your Personal Therapeutic AI Assistant. I'm here to listen."
            st.session_state.intro_given = True
            
    # Logic: Handling Sadness/Empathy
    elif any(word in user_text for word in ["sad", "not well", "bad", "depressed", "struggling"]):
        response = "I'm truly sorry to hear that you're feeling this way. Would you like to talk more about what's on your mind?"
        
    # Logic: General Greetings
    elif any(word in user_text for word in ["hi", "hello", "hey"]):
        if st.session_state.user_name:
            response = f"Hello again, {st.session_state.user_name}! How can I help you right now?"
        else:
            response = "Hello! I'm Pandora. How are you feeling today?"

    # Default Response
    else:
        responses = [
            "I'm listening. Please tell me more.",
            "I see. How does that make you feel?",
            "That's interesting. Can you elaborate on that?",
            "I'm here for you. What else is on your mind?"
        ]
        response = random.choice(responses)

    # 3. Display & Save Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
