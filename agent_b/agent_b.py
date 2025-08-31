from flask import Flask, request, jsonify
import requests
import os
from PIL import Image
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms

app = Flask(__name__)
AGENT_C_URL = "http://localhost:5003/generate-feedback"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======= X-RAY CHECKING CODE =======
xray_model = xrv.models.DenseNet(weights="densenet121-res224-all")
xray_model.eval()  # Set to inference mode

# Model expects normalized grayscale images
xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_diagnosis(image_path):
    img = Image.open(image_path).convert("L")
    img_tensor = xray_transform(img).unsqueeze(0)  # Shape: (1,1,224,224)

    with torch.no_grad():
        preds = xray_model(img_tensor)[0]
        probs = torch.sigmoid(preds)

    results = []
    for i, prob in enumerate(probs):
        if prob >= 0.5:
            results.append(f"{xray_model.pathologies[i]} ({prob:.2f})")

    if not results:
        return "✅ No findings with confidence ≥ 0.5."
    return "⚠️ Findings: " + ", ".join(results)

# ======= END X-RAY CODE =======

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    detected_type = request.form.get('image_type', 'Unknown')

    model_used = "TBD"
    analysis_result = "TBD"

    if detected_type.lower() == "x-ray":
        try:
            analysis_result = predict_diagnosis(image_path)
            model_used = "DenseNet121 (torchxrayvision)"
        except Exception as e:
            analysis_result = f"Error during X-ray analysis: {str(e)}"

    agent_b_response = {
        "agent_a_result": {"image_type": detected_type},
        "model_used": model_used,
        "analysis_result": analysis_result,
        "message_to_agent_c": "Prepared for Agent C"
    }

    try:
        response_c = requests.post(AGENT_C_URL, json=agent_b_response, timeout=100)
        if response_c.ok:
            agent_b_response["agent_c_feedback"] = response_c.json()
        else:
            agent_b_response["agent_c_feedback"] = {
                "error": f"Agent C returned status code {response_c.status_code}"
            }
    except requests.exceptions.RequestException as e:
        agent_b_response["agent_c_feedback"] = {"error": str(e)}

    return jsonify(agent_b_response), 200

if __name__ == "__main__":
    app.run(debug=True, port=5002, threaded=True)
