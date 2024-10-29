import nltk
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
except:
    print("NLTK data already downloaded")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str


class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self.knowledge_base = {
            "greetings": [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
                "howdy",
                "what's up",
            ],
            "farewells": [
                "goodbye",
                "bye",
                "see you",
                "take care",
                "good night",
                "catch you later",
                "farewell",
            ],
            "questions": [
                "how are you",
                "what do you do",
                "who are you",
                "what is your name",
                "what can you do",
            ],
            "responses": {
                "greetings": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Nice to meet you!",
                ],
                "farewells": [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Bye! Looking forward to our next chat!",
                ],
                "questions": [
                    "I'm a simple chatbot here to help you!",
                    "I'm doing great! How can I assist you?",
                    "I'm a friendly AI assistant ready to chat!",
                ],
                "unknown": [
                    "I'm not sure I understand. Could you rephrase that?",
                    "I'm still learning and don't quite understand. Can you try asking differently?",
                    "That's a bit beyond my current abilities. Could you try something simpler?",
                ],
            },
        }

        self.vectorizer = TfidfVectorizer()
        self._prepare_training_data()

    def _prepare_training_data(self):
        """Prepare and vectorize training data"""
        training_texts = []
        self.categories = []

        for category in ["greetings", "farewells", "questions"]:
            for phrase in self.knowledge_base[category]:
                training_texts.append(phrase)
                self.categories.append(category)

        self.train_vectors = self.vectorizer.fit_transform(training_texts)

    def _preprocess_text(self, text):
        """Preprocess input text"""
        tokens = word_tokenize(text.lower())

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]

        return " ".join(tokens)

    def _classify_intent(self, text):
        """Classify user input intent using cosine similarity"""
        preprocessed_text = self._preprocess_text(text)
        input_vector = self.vectorizer.transform([preprocessed_text])

        similarities = cosine_similarity(input_vector, self.train_vectors)

        if similarities.max() > 0.3:
            max_sim_idx = similarities.argmax()
            return self.categories[max_sim_idx]
        return "unknown"

    def get_response(self, user_input):
        """Generate response based on user input"""
        intent = self._classify_intent(user_input)

        responses = self.knowledge_base["responses"][intent]
        return np.random.choice(responses)


chatbot = Chatbot()


@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = chatbot.get_response(message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
