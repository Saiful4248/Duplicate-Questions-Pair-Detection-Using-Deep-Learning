import torch
from model import DuplicateQuestionDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description='Duplicate Question Detection CLI')
    parser.add_argument('--model_path', type=str, default='bert_model.pth', help='Path to saved model weights')
    parser.add_argument('--question1', type=str, required=True, help='First question')
    parser.add_argument('--question2', type=str, required=True, help='Second question')
    
    args = parser.parse_args()
    
    # Initialize model
    detector = DuplicateQuestionDetector(model_path=args.model_path)
    
    # Make prediction
    result = detector.predict(args.question1, args.question2)
    
    # Print results
    print(f"\nQuestion 1: {args.question1}")
    print(f"Question 2: {args.question2}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['duplicate_probability']*100:.2f}%")

if __name__ == "__main__":
    main()
