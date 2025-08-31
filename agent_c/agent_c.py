from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__)

# ===== LLM SETUP =====
MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. GPU not detected.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"✅ Loading model '{MODEL_ID}' on GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
# ======================

def generate_radiologist_feedback(image_type, analysis_result, model_used):
    """
    Generate patient-friendly feedback.
    Summarizes findings as: Normal/Healthy vs. Needs attention.
    """
    prompt = (
        f"You are a professional radiologist.\n"
        f"Image type: {image_type}\n"
        f"Analysis result: {analysis_result}\n"
        f"Model used: {model_used}\n\n"
        "Write a clear, patient-friendly summary:\n"
        "- Start with 'You are healthy in the following areas:' if some findings are normal.\n"
        "- Then say 'Please monitor or consult for:' for any detected abnormalities.\n"
        "- Do NOT repeat medical definitions of each disease.\n"
        "- Keep it concise and understandable for a patient.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,  # shorter and concise
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    for line in prompt.splitlines():
        full_output = full_output.replace(line, "")
    
    reply = full_output.strip()
    return reply


@app.route('/generate-feedback', methods=['POST'])
def generate_feedback():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    agent_a_result = data.get('agent_a_result', {})
    image_type = agent_a_result.get('image_type', 'Unknown')
    analysis_result = data.get('analysis_result', 'TBD')
    model_used = data.get('model_used', 'TBD')

    try:
        medical_feedback = generate_radiologist_feedback(image_type, analysis_result, model_used)
    except Exception as e:
        return jsonify({'error': f'LLM generation failed: {str(e)}'}), 500

    response = {
        "image_type": image_type,
        "analysis_result": analysis_result,
        "model_used": model_used,
        "medical_feedback": medical_feedback
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True, port=5003, threaded=True)
