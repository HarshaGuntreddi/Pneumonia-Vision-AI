#!/bin/bash
echo "Starting the training pipeline..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the training script
echo "Executing src/train.py..."
python src/train.py

echo "Training pipeline finished."