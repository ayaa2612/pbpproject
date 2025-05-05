import random
import re
import string
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources if not already done
# nltk.download('punkt')s
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# Load content from the text file
with open(''
          'medical research.txt', 'r', errors='ignore') as f:
    content = f.read()


# Preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)  # Return processed tokens as a string


# Preprocess content and prepare sentence tokens
preprocessed_content = preprocess_text(content)
sentence_tokens = nltk.sent_tokenize(content)

# Initialize TF-IDF Vectorizer with a custom tokenizer
# Removed `stop_words='english'` to avoid inconsistency warnings
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentence_tokens)

# Chatbot responses for greetings
greeting_inputs = ('hi', 'hello', 'what’s up', 'how are you?')
greeting_responses = [
    "Hello! I'm here to assist you with your medical concerns. How can I help today?",
    "Hi there! Welcome to our health support system. Please share a bit about your symptoms or history.",
    "Good day! How are you feeling today? I’m here to listen and provide guidance.",
    "Hello! I’m here to help you find the best care possible. How can I support you today?",
    "Hi! Tell me about your health concerns, and I’ll suggest some protocols for you."
]


# Function to handle greetings
def greet(user_input):
    for word in user_input.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None


# Function to generate chatbot response
def generate_response(user_response):
    user_response = preprocess_text(user_response)  # Preprocess the user response
    sentence_tokens.append(user_response)
    tfidf = vectorizer.transform(sentence_tokens)
    cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare with all other sentences

    idx = cosine_sim.argsort()[0][-1]  # Get the index of the most similar sentence
    flat = cosine_sim.flatten()
    flat.sort()
    score = flat[-2]  # Get the similarity score of the best match

    if score == 0:  # No good response found
        response = "I'm sorry, I didn't understand that. Could you please elaborate?"
    else:
        response = sentence_tokens[idx]

    sentence_tokens.pop(-1)  # Remove user input from sentence tokens
    return response


# Flask route for home
@app.route('/')
def home():
    return render_template('index.html')


# Flask route for chatbot response
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    user_input = user_input.lower()

    if user_input == 'bye':
        return "Goodbye! Have a great day!"

    greeting_response = greet(user_input)
    if greeting_response:
        return greeting_response

    return generate_response(user_input)


if __name__ == "__main__":
    app.run(debug=True)





