from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

AGENT_C_URL = "http://localhost:5003/generate-feedback"

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Agent B:
    - Receives image and detected type from Agent A
    - Prepares a structured response
    - Calls Agent C to get medical feedback synchronously
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read detected image type
    detected_type = request.form.get('image_type', 'Unknown')

    # TODO: Implement model analysis here
    model_used = "TBD"
    analysis_result = "TBD"

    # Prepare Agent B response
    agent_b_response = {
        "agent_a_result": {"image_type": detected_type},
        "model_used": model_used,
        "analysis_result": analysis_result,
        "message_to_agent_c": "Prepared for Agent C"
    }

    # Call Agent C synchronously
    try:
        response_c = requests.post(AGENT_C_URL, json=agent_b_response, timeout=10)
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
