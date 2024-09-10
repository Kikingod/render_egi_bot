from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import core.agent_tools
from core.run_agent import get_ai_response, get_full_response, get_user_responses
import traceback

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]}}, supports_credentials=True)



acctivatation_message = "Look up in your docs if you epomaker ships internationally"

@app.route('/api/chat', methods=['POST']) 
def chat():
    thread_id = 201
    try:
        user_message = request.json.get('message')
        print(f'Received message: {user_message}')
        
        response = get_user_responses([user_message], thread_id=thread_id)

        print(f'Response type: {type(response)}')
        print(f'Response content: {response}')

        print(f'Sending response: {response}')
        return jsonify({'response': response})
    except Exception as e:
        print(f'Error occurred: {e}')
        print(f'Error traceback: {traceback.format_exc()}')
        return jsonify({'response': 'An error occurred'}), 500
    
@app.route('/', methods= ['GET', 'POST'])
def home():
    return render_template('main.html')

def retrieve_info(query):
    # Placeholder function to simulate retrieval tool usage
    return "Retrieved information based on query: " + query

if __name__ == '__main__':
    app.run(debug=True)
