import torch
import requests
from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from io import BytesIO
from PIL import Image

print(torch.version.cuda)     
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
AGENT_B_URL = "http://localhost:5002/process-image"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

try:
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    model = AutoModelForZeroShotImageClassification.from_pretrained("google/medsiglip-448").to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    processor = None
    model = None

# Trusted types dictionary
ALLOWED_IMAGE_TYPES = {"X-ray", "CT scan", "Ultrasound"}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            image = Image.open(file.stream).convert("RGB")
            detected_type = "Unknown"

            if model and processor:
                candidate_labels = ["X-ray", "MRI", "CT scan", "Ultrasound", "Dermoscopy",
                                    "Microscopy", "Angiogram", "Mammogram", "PET scan"]

                image_inputs = processor(images=image, return_tensors="pt").to(device)
                text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True).to(device)

                with torch.no_grad():
                    outputs = model(**image_inputs, **text_inputs)
                    logits = outputs.logits_per_image

                predicted_label = candidate_labels[logits.argmax().item()]
                # Filter by allowed dictionary
                detected_type = predicted_label if predicted_label in ALLOWED_IMAGE_TYPES else "Unknown"

            else:
                return jsonify({'error': 'Model or processor not loaded correctly'}), 500

            file.stream.seek(0)

            files = {'image': (file.filename, file.stream, file.mimetype)}
            data = {'image_type': detected_type}

            response = requests.post(AGENT_B_URL, files=files, data=data, timeout=150)
            if response.status_code != 200:
                return jsonify({'error': f"Agent B returned status code {response.status_code}"}), 500

            return jsonify(response.json()), 200

        except requests.exceptions.ConnectionError:
            return jsonify({'error': "Cannot connect to Agent B"}), 500
        except requests.exceptions.Timeout:
            return jsonify({'error': "Agent B request timed out"}), 500
        except Exception as e:
            return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
