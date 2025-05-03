# Mini-Transformer: Language Model From Scratch

Hey, Im Ashwin. This is a small (10M-30M parameter) Transformer-based language model from scratch using PyTorch's low-level tensor operations. The model is trained on the WikiText-2 dataset for next-token prediction. I am currently building a "Mini-GPT" for this, as I think a frontend would be very valuable. Stay Tuned!

## Features

- Custom implementation of the Transformer architecture
- Multi-head self-attention with masking
- Layer normalization and residual connections
- Position-wise feedforward networks
- Positional encodings
- Training loop with gradient clipping and AdamW optimizer
- Learning rate scheduler with warmup

## Setup and Training

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download and preprocess the WikiText-2 dataset:
   ```
   python src/data/download_data.py
   ```

3. Train the model:
   ```
   python src/train/train.py
   ```

4. Evaluate the model:
   ```
   python src/train/evaluate.py
   ```

## Model Configuration

The model size and architecture can be configured in `src/model/config.py`. 
