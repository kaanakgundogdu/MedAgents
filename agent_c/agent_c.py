from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate-feedback', methods=['POST'])
def generate_feedback():
    """
    Agent C:
    - Receives data from Agent B (image type, analysis results, model used)
    - Provides medical feedback quickly
    """
    data = request.get_json(force=True)  # Faster, forces JSON parsing
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Extract info from Agent B
    agent_a_result = data.get('agent_a_result', {})
    image_type = agent_a_result.get('image_type', 'Unknown')
    analysis_result = data.get('analysis_result', 'TBD')
    model_used = data.get('model_used', 'TBD')

    # TODO: Main logic for generating meaningful feedback
    medical_feedback = f"TBD: Advice based on {image_type} using {model_used}"

    # Minimal dictionary construction
    response = {
        "image_type": image_type,
        "analysis_result": analysis_result,
        "model_used": model_used,
        "medical_feedback": medical_feedback
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True, port=5003, threaded=True)
