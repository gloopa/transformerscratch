#!/bin/bash
set -e

# Create and activate a Python virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies one by one to avoid issues
echo "Installing dependencies..."
pip install torch>=1.9.0
pip install numpy>=1.20.0
pip install tqdm>=4.61.0
pip install matplotlib>=3.4.0
pip install datasets>=1.8.0
pip install tokenizers>=0.10.0

# Skip the data download and preprocessing step for testing
echo "Skipping data download for testing..."
# python src/data/download_data.py --dataset wikitext-2-raw-v1 --output_dir data

# Train the model
echo "Training the model..."
python src/train/train.py \
    --data_dir data/wikitext-2 \
    --output_dir outputs \
    --checkpoint_dir checkpoints \
    --plots_dir plots \
    --hidden_size 384 \
    --num_layers 6 \
    --num_heads 6 \
    --intermediate_size 1536 \
    --num_epochs 3 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --scheduler cosine \
    --mixed_precision

# Evaluate the model
echo "Evaluating the model..."
python src/train/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/wikitext-2 \
    --generate \
    --prompt "The history of artificial intelligence" \
    --max_length 200 \
    --top_k 50 \
    --top_p 0.9 \
    --temperature 0.8 \
    --do_sample \
    --output_file generated_text.txt

echo "Pipeline completed successfully!" 