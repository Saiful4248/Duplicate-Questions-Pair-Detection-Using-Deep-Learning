import torch
from transformers import BertTokenizer, BertForSequenceClassification

class DuplicateQuestionDetector:
    def __init__(self, model_path=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Load saved weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using default BERT weights")
        
        # Set to evaluation mode
        self.model.eval()
    
    def tokenize_questions(self, q1, q2, max_length=128):
        return self.tokenizer.encode_plus(
            q1,
            text_pair=q2,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
    
    def predict(self, q1, q2):
        # Tokenize
        encoded = self.tokenize_questions(q1, q2)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        attention_mask = encoded['attention_mask']
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            duplicate_probability = probabilities[0][1].item()
        
        result = {
            'is_duplicate': bool(predicted_class),
            'duplicate_probability': duplicate_probability,
            'prediction': "Duplicate" if predicted_class == 1 else "Not Duplicate"
        }
        
        return result
