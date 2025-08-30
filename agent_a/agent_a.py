from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
AGENT_B_URL = "http://localhost:5002/process-image"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # ---------------------------
        # Step 1: Detect image type
        # ---------------------------
        detected_type = 'X-ray'  # Dummy detection logic

        try:
            # ---------------------------
            # Step 2: Send to Agent B in memory
            # ---------------------------
            file_stream = BytesIO(file.read())
            files = {'image': (file.filename, file_stream, file.mimetype)}
            data = {'image_type': detected_type}

            response = requests.post(AGENT_B_URL, files=files, data=data, timeout=15)

            if response.status_code != 200:
                return jsonify({'error': f"Agent B returned status code {response.status_code}"}), 500

            # Return Agent B's response directly
            return jsonify(response.json()), 200

        except requests.exceptions.ConnectionError:
            return jsonify({'error': "Cannot connect to Agent B"}), 500
        except requests.exceptions.Timeout:
            return jsonify({'error': "Agent B request timed out"}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
