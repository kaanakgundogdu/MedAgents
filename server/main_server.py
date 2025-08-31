from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return app.send_static_file('main_page.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Send request to Agent A directly from memory
        agent_a_response = send_image_to_agent_a(file)
        if 'error' in agent_a_response:
            return jsonify(agent_a_response), 500
        
        # Merge Agent A's response with final feedback if Agent C returned it
        response_payload = agent_a_response.copy()
        final_feedback = agent_a_response.get('agent_c_feedback', {}).get('medical_feedback')
        if final_feedback:
            response_payload['final_feedback'] = final_feedback
        
        return jsonify(response_payload), 200
    
    return jsonify({'message': 'Invalid file type'}), 400

def send_image_to_agent_a(file):
    agent_a_url = 'http://localhost:5001/detect-image'
    
    try:
        # Read file into memory
        file_stream = BytesIO(file.read())
        files = {'image': (file.filename, file_stream, file.mimetype)}
        
        response = requests.post(agent_a_url, files=files, timeout=150)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f"Error communicating with Agent A (status code {response.status_code})"}
    except requests.exceptions.ConnectionError:
        return {'error': "Error: Agent A is not running on port 5001"}
    except requests.exceptions.Timeout:
        return {'error': "Error: Agent A did not respond in time"}

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
