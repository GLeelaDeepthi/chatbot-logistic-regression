# Importing required libraries
import nltk
import numpy as np
import random
import json
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Data: Sample intents (Tag, Patterns, Responses)
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you?", "Good morning", "Hey"],
            "responses": ["Hello!", "Hi there!", "How can I assist you today?", "Good morning!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Take care"],
            "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!"]
        },
        {
            "tag": "age",
            "patterns": ["How old are you?", "What is your age?", "Tell me your age"],
            "responses": ["I am a bot, I don't age!", "Age is just a number for me!", "I don't have an age."]
        },
        {
            "tag": "name",
            "patterns": ["What is your name?", "Tell me your name", "Who are you?"],
            "responses": ["I am your chatbot!", "I don't have a specific name, but you can call me ChatBot."]
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "Thanks a lot"],
            "responses": ["You're welcome!", "Glad I could help!", "Happy to assist!"]
        }
    ]
}

# Preprocess the data
def preprocess_data(intents):
    patterns = []
    tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    return patterns, tags

# Prepare data for training the model
patterns, tags = preprocess_data(intents)

# Vectorize the training data (convert text to a feature matrix)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(patterns).toarray()

# Encode tags as integers
tag_to_index = {tag: i for i, tag in enumerate(set(tags))}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}
y_train = np.array([tag_to_index[tag] for tag in tags])

# Training the Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Function to get the response based on the model's prediction
def get_response(user_input):
    # Convert user input into feature matrix using the same vectorizer
    bow_input = vectorizer.transform([user_input]).toarray()
    prediction = classifier.predict(bow_input)
    tag = index_to_tag[prediction[0]]

    # Find the response for the predicted tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

# Streamlit web app interface
def chat_interface():
    st.title("Chatbot with Logistic Regression")
    user_input = st.text_input("You: ", "")

    if user_input:
        response = get_response(user_input)
        st.write(f"Bot: {response}")

# Running the Streamlit web app
if __name__ == "__main__":
    chat_interface()
