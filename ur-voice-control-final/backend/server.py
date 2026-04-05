import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import sys

from chatbot import Chatbot
from ur_agent import URAgent
from voice_model import detectV2

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

chatbot = Chatbot()
ur_agent = URAgent()

@app.route('/')
def hello():
    return 'running agent backend'

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message")
    print(msg)
    response = ur_agent.invoke(msg)
    return jsonify({'response': response})

@app.route("/patch", methods=["GET"])
def patch():
    word = asyncio.run(detectV2.predict_once())
    response = ur_agent.invoke(word)
    
    return jsonify({'response': response})

# voice control route - using open ai whisper?
@app.route("/talk", methods=["POST"])
def talk():
    return jsonify({'response': "temp"})

if __name__ == "__main__":
    app.run(port=5555, debug=True)  