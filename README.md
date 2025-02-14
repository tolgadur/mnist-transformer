# MNIST Transformer

This educational project implements a Transformer-based architecture for processing MNIST digit sequences, along with a single-digit classifier. I did this as part of the Machine Learning Institute to really get into the details of the Transformer architecture, by implementing it from scratch. The project consists of two main components:

1. A sequence-to-sequence transformer model for predicting digit sequences
2. A simpler single-digit classifier that serves as a sanity check for the encoder's ability to process MNIST images effectively

## Project Structure

```
mnist-transformer/
├── src/
│   ├── attention.py      # Attention mechanism implementation
│   ├── classifier.py     # Single digit classification model
│   ├── config.py         # Configuration settings
│   ├── dataset.py        # MNIST dataset loaders
│   ├── decoder.py        # Transformer decoder
│   ├── encoder.py        # Transformer encoder
│   ├── evals.py         # Evaluation utilities
│   ├── main.py          # Main entry point
│   ├── mlp.py           # MLP implementation
│   ├── train.py         # Training loops
│   ├── transformer.py    # Full transformer model
│   └── utils.py         # Utility functions
├── models/              # Saved model checkpoints
└── README.md
```

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd mnist-transformer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

1. Train the Transformer model for sequence prediction:

```python
from src.train import train_transformer

# Train with default parameters
train_transformer()

# Or customize training parameters
train_transformer(
    epochs=30,
    batch_size=32,
    lr=0.001
)
```

2. Train the single digit classifier:

```python
from src.train import train_single_digit_classifier

train_single_digit_classifier(
    epochs=10,
    batch_size=32,
    lr=0.001
)
```

### Model Architecture

#### Transformer Model

- Uses a patch-based encoder for processing MNIST images
- Implements a decoder with multi-head attention
- Includes gradient clipping and learning rate scheduling
- Uses label smoothing in the loss function
- Model dimensionality: 64
- Number of attention heads: 4

#### Single Digit Classifier

- Serves as a validation tool for the encoder's image processing capabilities
- Processes flattened image patches using the same encoder architecture
- Simpler task (single digit classification) to verify the encoder design
- Uses CrossEntropyLoss for training

## Model Checkpoints

The trained models are saved in the `models/` directory:

- Transformer model: `models/transformer_model.pth`
- Classifier model: `models/classifier_single_digit_model.pth`

## Training Parameters

### Transformer

- Default epochs: 30
- Batch size: 32
- Learning rate: 0.001
- Dropout: 0.1
- Label smoothing: 0.1
- Gradient clipping: 1.0
- Learning rate scheduler: CosineAnnealingLR

### Classifier

- Default epochs: 10
- Batch size: 32
- Learning rate: 0.001
