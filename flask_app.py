from flask import Flask, render_template, request, jsonify
from model import DuplicateQuestionDetector
import os

app = Flask(__name__)

# Initialize model
model_path = "bert_model.pth"
detector = DuplicateQuestionDetector(model_path=model_path if os.path.exists(model_path) else None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    q1 = data.get('question1', '')
    q2 = data.get('question2', '')
    
    if not q1 or not q2:
        return jsonify({'error': 'Both questions are required'}), 400
    
    result = detector.predict(q1, q2)
    
    return jsonify({
        'prediction': result['prediction'],
        'is_duplicate': result['is_duplicate'],
        'confidence': round(result['duplicate_probability'] * 100, 2)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
