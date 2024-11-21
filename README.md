# Question Generation in Portuguese

This project implements and evaluates different T5-based models for question generation in Portuguese. The system can generate questions (and optionally answers) from given context passages using various pre-trained models.

## Features

- Support for multiple T5-based models:
  - PTT5 (small, base, large)
  - FLAN-T5 (small, base, large)
- Multiple dataset support:
  - PIRA dataset
  - FairytaleQA (Portuguese)
  - SQuAD v2 (Portuguese)
- Configurable input/output modes:
  - Context-only input → Question generation
  - Context + Answer input → Question generation
  - Context input → Question + Answer generation
- Comprehensive evaluation metrics:
  - ROUGE (1, 2, L)
  - BERTScore
  - BLEU
  - METEOR

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Datasets
- NLTK
- Evaluate
- BERTScore
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/laicsiifes/question_generation_pt.git
cd question_generation_pt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning Models

Use `run_fine_tuning.py` to train the models:

```bash
python run_fine_tuning.py
```

Key configuration options in the script:
- `dataset_name`: Choose between 'pira', 'fairytale_pt_qa', or 'squad_pt_v2'
- `model_name`: Select model type ('ptt5_small', 'ptt5_base', 'ptt5_large', 'flan_t5_small', 'flan_t5_base', 'flan_t5_large')
- `use_answer_input`: Whether to include answers in input (Boolean)
- `output_with_answer`: Whether to generate both questions and answers (Boolean)
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size

### Evaluating Models

Use `run_eval_models.py` to evaluate trained models:

```bash
python run_eval_models.py
```

The script evaluates all trained models on the test set and produces:
- CSV files with evaluation metrics
- JSON files with detailed predictions

## Project Structure

```
├── data/                    # Data directory for models and results
├── src/
│   ├── models_utils.py      # Utility functions for model processing
│   └── evaluation_measures.py # Implementation of evaluation metrics
├── run_fine_tuning.py       # Script for model fine-tuning
├── run_eval_models.py       # Script for model evaluation
└── requirements.txt         # Project dependencies
```

## Model Configuration Options

### Input Modes
1. Context-only:
   - Input: Just the context passage
   - Output: Generated question
   
2. Context + Answer:
   - Input: Context passage and target answer
   - Output: Generated question

3. Context → Question + Answer:
   - Input: Context passage
   - Output: Both question and answer

### Model Parameters

- Input max length: 512 tokens
- Output max length: 
  - Question only: 40 tokens
  - Question + Answer: 120 tokens
- Training batch sizes:
  - Small/Base models: 16
  - Large models: 4
  - Adjusted for question + answer generation

## Evaluation Metrics

The system evaluates generated questions (and answers when applicable) using:
- ROUGE scores (1, 2, L) for measuring text overlap
- BERTScore for semantic similarity
- BLEU score for translation quality
- METEOR score for semantic adequacy

Results are saved in CSV format for questions and answers separately.

## Citation

...to come

## Contributors

Tiago Felipe Vivaldi Braga, Hilário Tomaz Alves de Oliveira and Bruno Cardoso Coutinho.

## Acknowledgments

- PTT5 models from Unicamp-DL
- FLAN-T5 models from Google
- Dataset providers: PIRA, FairytaleQA, SQuAD v2
