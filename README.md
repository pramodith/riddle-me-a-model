# Riddle Me A Model

A project for training and evaluating a language model to solve and reason about riddles using custom reward functions and a chain-of-thought (CoT) format with XML tags.

## Features
- **Custom Trainer**: Uses a `RiddleTrainer` class for model training and evaluation, leveraging LoRA and vLLM for efficient fine-tuning.
- **Reward Functions**: Multiple reward functions for answer correctness, answer length, reasoning (think) length, and XML format compliance.
- **Dataset Utilities**: Functions to load, split, and filter the riddles dataset.
- **Configurable Prompts**: Uses a system prompt and XML-based CoT format for model responses.

## Project Structure
- `src/grpo_trainer.py`: Main training and evaluation logic. Defines `RiddleTrainer` for model setup, training, and test evaluation.
- `src/reward_functions.py`: Implements reward functions for reinforcement learning from human feedback (RLHF), including correctness, answer length, think length, and XML compliance.
- `src/process_data.py`: Functions to load the riddles dataset, split it into train/dev/test, and filter by difficulty.
- `src/constants.py`: Contains prompt templates and XML formatting constants.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd riddle-me-a-model
   ```
2. **Install dependencies:**
   - Python 3.12+
   - Install core dependencies:
     ```bash
     pip install uv
     uv sync
     ```

## Usage
1. **Prepare the dataset:**
   - The code expects a dataset named `Pramodith/riddles_dataset_scored` (Hugging Face Datasets).
2. **Run training:**
   ```bash
   python src/grpo_trainer.py
   ```
   This will train the model and evaluate it on the test set.

## Configuration
- Edit `src/constants.py` to change the system prompt or XML format.
- Adjust hyperparameters in `RiddleTrainer` or via the script arguments as needed.

## Main Dependencies
- [trl](https://github.com/huggingface/trl)
- [unsloth](https://github.com/unslothai/unsloth)
- [datasets](https://github.com/huggingface/datasets)
- [scikit-learn](https://scikit-learn.org/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

## License
Add your license here.

## Acknowledgements
- Inspired by chain-of-thought prompting and reinforcement learning from human feedback (RLHF) research.
- Uses [Pramodith/riddles_dataset_scored](https://huggingface.co/datasets/Pramodith/riddles_dataset_scored).
