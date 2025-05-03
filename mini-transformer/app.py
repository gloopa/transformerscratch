from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
import os
import json
import time
from src.model.transformer import MiniTransformer
from src.model.config import MiniTransformerConfig
from src.data.tokenizer import TransformerTokenizer

app = Flask(__name__)

# Load model and tokenizer
def load_model():
    checkpoint_path = os.path.join('checkpoints', 'final_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    
    # Create config object
    config = MiniTransformerConfig(
        vocab_size=config_dict.get('vocab_size', 50257),
        hidden_size=config_dict.get('hidden_size', 128),
        num_hidden_layers=config_dict.get('num_hidden_layers', 4),
        num_attention_heads=config_dict.get('num_attention_heads', 4),
        intermediate_size=config_dict.get('intermediate_size', 512),
        hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=config_dict.get('attention_probs_dropout_prob', 0.1),
        max_position_embeddings=config_dict.get('max_position_embeddings', 1024)
    )
    
    # Initialize model with config
    model = MiniTransformer(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer from data directory
    tokenizer_path = os.path.join('data', 'wikitext', 'tokenizer')
    tokenizer = TransformerTokenizer(tokenizer_path)
    
    return model, tokenizer

# Initialize model and tokenizer
model, tokenizer = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 50))
    temperature = float(data.get('temperature', 0.7))
    
    # Return a streaming response
    return Response(stream_with_context(generate_stream(prompt, max_tokens, temperature)),
                   content_type='text/event-stream')

def generate_stream(prompt, max_tokens, temperature):
    """Stream the generated tokens one by one."""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # Generate text token by token
    with torch.no_grad():
        for i in range(max_tokens):
            # Get model predictions for the next token
            outputs = model(input_tensor)
            logits = outputs['logits']
            
            # Get the predictions for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Decode the current token and yield it
            decoded_token = tokenizer.decode([next_token])
            yield f"data: {json.dumps({'token': decoded_token})}\n\n"
            
            # Small delay to make the typing effect visible
            time.sleep(0.05)
            
            # Append the token to the input for next iteration
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)
            
            # Check if we've reached an end condition
            if next_token == tokenizer.eos_token_id:
                break
    
    # Signal completion
    yield f"data: {json.dumps({'done': True})}\n\n"

if __name__ == '__main__':
    app.run(debug=True) 