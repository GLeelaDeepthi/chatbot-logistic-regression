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
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

# Preprocess the data (tokenize, remove punctuation, etc.)
def preprocess_data(intents):
    patterns = []
    tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    
    return patterns, tags

# Tokenize and clean up the patterns
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words if word not in string.punctuation]
    return sentence_words

# Bag of words model
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return(np.array(bag))

# Prepare data for training the model
patterns, tags = preprocess_data(intents)
words = sorted(list(set([word for pattern in patterns for word in clean_up_sentence(pattern)])))

# Create training data
training_sentences = patterns
training_labels = [tags.index(tag) for tag in tags]

# Vectorize the training data (convert text to a feature matrix)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences).toarray()

# Training the Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train, training_labels)

# Function to get the response based on the model's prediction
def get_response(user_input):
    # Convert user input into bag of words
    bow_input = bow(user_input, words)
    prediction = classifier.predict([bow_input])
    tag = tags[prediction[0]]
    
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
