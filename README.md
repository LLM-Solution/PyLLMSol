# Python Tool Box for [LLM Solutions](https://llm-solutions.fr)

![Pylint](https://github.com/LLM-Solution/PyLLMSol/actions/workflows/pylint.yml/badge.svg)

**PyLLMSol** is a Python package designed to simplify the training and inference processes for **large language models (LLMs)**. With dedicated modules for both **training** and **inference**, **PyLLMSol** allows users to create checkpoints, manage prompts, and run models using CLI or API interfaces.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Structure](#structure)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [License](#license)
- [Author](#author)

## Installation

To install **PyLLMSol** and its dependencies, follow these steps:

### 1. Install PyTorch

First, install **PyTorch** with GPU or CPU support by following the instructions on the [PyTorch official website](https://pytorch.org/get-started/locally/). Choose the appropriate command for your operating system, Python version, and hardware.

### 2. Clone the repository

Clone the **PyLLMSol** repository to your local machine:

```bash
git clone https://github.com/LLM-Solution/PyLLMSol.git
cd PyLLMSol
```

### 3. Install dependencies

Install the required Python packages listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Install PyLLMSol

Install the **PyLLMSol** package using pip:

```bash
pip install .
```

## Features

- **Training Management**: Handle datasets, manage training steps, and track losses.
- **Checkpointing**: Save and load checkpoints of models and data at regular intervals.
- **CLI Interface**: Interact with the model via command-line.
- **API Support**: Host the model as a REST API with Flask.
- **Prompt Management**: Handle prompts with truncation and formatting options.


## Structure

```plaintext
PyLLMSol/
├── setup.py
├── requirements.txt
├── pyllmsol/
│   └── argparser.py
│   └── _base.py                 # Basis of training and inference modules
│   ├── data/                    # Data module
│   │   └── _base_data.py
│   │   └── chat.py              # Chat objects with LLaMa-3.2 format
│   │   └── prompt.py
│   │   └── utils.py
│   └── inference/               # Inference module
│   │   └── _base_api.py
│   │   └── _base_cli.py
│   │   └── cli_instruct.py      # CLI with LLaMa-3.2 chat format
│   └── training/                # Training module
│   │   └── checkpoint.py
│   │   └── instruct_trainer.py  # Trainer with LLaMa-3.2 chat format
│   │   └── loss.py
│   │   └── trainer.py
│   │   └── utils.py
│   └── test/
└── README.md
```

### 1. `training` Module

The `training` module contains tools for managing training workflows and model checkpoints.

- **Trainer**: Manages training loops, handles batch processing, and tracks loss over time.
- **Checkpoint**: Saves and loads model states and data at specific intervals, enabling easy restoration of the training process.
- **DataBrowser**: Supports batch processing and iterating over data with customizable parameters.

#### Example Usage

```python
from pyllmsol.training import Trainer, Checkpoint

# Initialize training components
trainer = Trainer(llm=my_model, tokenizer=my_tokenizer, dataset=my_data, batch_size=16)
checkpoint = Checkpoint(path='./checkpoints')

# Run training with checkpointing
trainer.run(device='cuda', checkpoint=checkpoint)
```

### 2. `inference` Module

The `inference` module supports generating responses from the model and includes both CLI and API options.

- **_BaseCommandLineInterface**: Offers an interactive command-line interface for chatting with the model.
- **API**: Provides a REST API using Flask, allowing remote model access.

#### CLI Usage

Run the command-line interface to interact with your LLM:

```bash
python -m pyllmsol.inference._base_cli --model_path path/to/model
```

#### API Usage

To launch the API:

```python
from pyllmsol.inference import API

api = API(model_path='path/to/model', init_prompt='Hello! How can I assist you?')
api.run(host="0.0.0.0", port=5000)
```

## Dependencies

PyLLMSol requires Python 3.10 or later. Core dependencies include:

- `flask>=3.0.3`
- `llama-cpp-python>=0.3.1`
- `matplotlib>=3.9.2`
- `pandas>=2.2.3`
- `peft>=0.13.2`
- `sentencepiece>=0.2.0`
- `torch>=2.5.0`
- `transformers>=4.45.2`
- `tqdm>=4.66.5`

For a full list, see requirements.txt.

## Getting Started

1. Clone the repository and install dependencies.
2. Set up your model files and ensure you have the necessary model weights.
3. Use the CLI or API modules to interact with the model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[LLM Solutions](https://llm-solutions.fr) - [Arthur Bernard](https://www.linkedin.com/in/arthur-bernard-789955152/) - contact@llm-solutions.fr

___

For further information, refer to the documentation in the source files.

