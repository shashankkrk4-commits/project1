import os
import sys
import json
import random
import threading
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import streamlit as st
import speech_recognition as sr
import pyttsx3

# --- 1. DOWNLOAD NLTK RESOURCES ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- 2. TEXT PREPROCESSING (NLP UTILITIES) ---
stemmer = PorterStemmer()

def tokenize(sentence):
    """Splits a sentence into words."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Finds the root form of a word."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """Encodes tokenized sentence into a binary bag-of-words array."""
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# --- 3. HARDCODED TRAINING DATASET (INTENTS) ---
# No external JSON file is required; everything is embedded here.
intents = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hey", "Howdy", "Is anyone there?", "Hello", "Good day", "What's up"],
      "responses": ["Hello! How can I help you today?", "Good to see you! How can I assist you?", "Hi! What can I do for you?"],
      "context_set": ""
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "I am leaving", "Talk to you later", "Quit"],
      "responses": ["Sad to see you go! Have a great day.", "Goodbye! Reach out anytime.", "Bye! Take care."],
      "context_set": ""
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thank you so much"],
      "responses": ["Happy to help!", "Anytime!", "My pleasure!", "You're very welcome!"]
    },
    {
      "tag": "book_ticket_start",
      "patterns": ["I want to book a ticket", "Can I book a seat?", "Ticket booking"],
      "responses": ["Sure, I can help with that! Which city are you traveling to?"],
      "context_set": "booking_destination"
    },
    {
      "tag": "book_ticket_destination",
      "patterns": ["To New York", "Destination is London", "Paris", "Tokyo", "Chicago"],
      "responses": ["Understood. Please provide your travel date."],
      "context_filter": "booking_destination",
      "context_set": "booking_date"
    },
    {
      "tag": "book_ticket_date",
      "patterns": ["Tomorrow", "Next Monday", "On 15th October", "25th December", "Today"],
      "responses": ["Perfect! Your booking request is registered. Is there anything else?"],
      "context_filter": "booking_date",
      "context_set": ""
    }
  ]
}

# --- 4. MODEL ARCHITECTURE (PyTorch) ---
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# --- 5. TRAINING FUNCTION ---
def train_model():
    """Trains the Neural Network on startup if model file does not exist."""
    print("Auto-training NLP model...")
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',', ';']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(list(set(all_words)))
    tags = sorted(list(set(tags)))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "model_data.pth"
    torch.save(data, FILE)
    print("Training finished! Saved model_data.pth")

# --- 6. AUTO-TRAIN ON STARTUP ---
MODEL_FILE = "model_data.pth"
if not os.path.exists(MODEL_FILE):
    train_model()

# --- 7. LOAD TRAINED MODEL ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(MODEL_FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# --- 8. AUDIO ENGINE UTILITIES ---
def speak_text(text):
    """Executes Text-to-Speech in an isolated background thread to keep UI smooth."""
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run_tts, daemon=True).start()

def listen_speech():
    """Captures microphone input and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Listening... Speak clearly!", icon="🎙️")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Audio not clear. Please try speaking again.")
        except sr.RequestError as e:
            st.error(f"Speech Service unavailable: {e}")
        except Exception as e:
            st.error(f"Error handling voice input: {e}")
    return None

# --- 9. STREAMLIT CHAT UI ---
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Chat Assistant")
st.subheader("Your Intelligent NLP-Powered Virtual Companion")

# Initialize persistent session states
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}]
if "context" not in st.session_state:
    st.session_state.context = ""
if "voice_output" not in st.session_state:
    st.session_state.voice_output = False

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.voice_output = st.checkbox("Enable Text-to-Speech (Voice Output)", value=st.session_state.voice_output)
    
    st.markdown("### 🎙️ Speech Input")
    if st.button("Click to Speak"):
        user_voice = listen_speech()
        if user_voice:
            st.session_state.voice_input_value = user_voice
            st.rerun()

    st.markdown("---")
    if st.button("Reset Chat Session", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}]
        st.session_state.context = ""
        st.rerun()

# --- 10. CONTEXT-AWARE RESPONSE CORE ---
def generate_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    outputs = model(X)
    _, predicted = torch.max(outputs, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(outputs, dim=1)
    prob = probs[0][predicted.item()]

    response = "I'm sorry, I don't understand that. Could you rephrase your question?"
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Match against Active Context filter
                if "context_filter" in intent and intent["context_filter"] != st.session_state.context:
                    continue
                
                response = random.choice(intent['responses'])
                
                # Update Context Session State
                if "context_set" in intent:
                    st.session_state.context = intent["context_set"]
                else:
                    st.session_state.context = ""
                break
    return response

# Display Previous Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture User Query (Either from voice input trigger or direct typing)
user_query = None

if "voice_input_value" in st.session_state and st.session_state.voice_input_value:
    user_query = st.session_state.voice_input_value
    del st.session_state.voice_input_value  # Clean up voice cache immediately

text_query = st.chat_input("Type your message here...")
if text_query:
    user_query = text_query

# Handle incoming query execution
if user_query:
    # Append & Display User text
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Process AI predictions and respond
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            bot_reply = generate_response(user_query)
            st.write(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
            # Fire Text-To-Speech if active
            if st.session_state.voice_output:
                speak_text(bot_reply)