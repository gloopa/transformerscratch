# Mini-Transformer: Language Model From Scratch

This project implements a small (10M-30M parameter) Transformer-based language model from scratch using PyTorch's low-level tensor operations. The model is trained on the WikiText-2 dataset for next-token prediction.

## Features

- Custom implementation of the Transformer architecture
- Multi-head self-attention with masking
- Layer normalization and residual connections
- Position-wise feedforward networks
- Positional encodings
- Training loop with gradient clipping and AdamW optimizer
- Learning rate scheduler with warmup
- Mixed precision training (optional)

## Project Structure

```
mini-transformer/
├── data/
│   └── wikitext-2/               # Tokenized & preprocessed WikiText data
├── src/
│   ├── model/                    # Model implementation
│   ├── train/                    # Training logic
│   └── data/                     # Data processing
├── plots/                        # Training visualizations
├── checkpoints/                  # Saved model checkpoints
├── README.md
└── requirements.txt
```

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