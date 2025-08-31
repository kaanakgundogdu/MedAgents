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
    findings = {}
    for item in analysis_result.split(','):
        item = item.strip()
        if '(' in item and ')' in item:
            name, val = item.split('(')
            name = name.strip()
            try:
                val = float(val.strip(')'))
                findings[name] = val
            except:
                continue

    threshold = 0.5  # probability threshold
    normal_findings = [k for k, v in findings.items() if v < threshold]
    abnormal_findings = [k for k, v in findings.items() if v >= threshold]

    abnormal_findings = sorted(abnormal_findings, key=lambda x: -findings[x])[:5]

    simple_summary = (
        f"Normal areas: {', '.join(normal_findings) if normal_findings else 'None'}\n"
        f"Abnormal areas needing attention: {', '.join(abnormal_findings) if abnormal_findings else 'None'}"
    )

    prompt = (
        f"You are a professional radiologist.\n"
        f"Image type: {image_type}\n"
        f"Summary of findings:\n{simple_summary}\n\n"
        "Write a short, patient-friendly report. "
        "- Start with 'You are healthy in the following areas:'\n"
        "- Then say 'Please monitor or consult for:'\n"
        "- Avoid repeating medical definitions.\n"
        "- Keep it concise and clear for a patient.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt content
    for line in prompt.splitlines():
        full_output = full_output.replace(line, "")
    
    return full_output.strip()


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
