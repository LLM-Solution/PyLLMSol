# Python Tool Box for LLM Solutions

PyLLMSol is a Python package designed to simplify the training and inference processes for large language models (LLMs). With dedicated modules for both training and inference, PyLLMSol allows users to create checkpoints, manage prompts, and run models using CLI or API interfaces.

## Installation

To install PyLLMSol, clone this repository and install the package using `pip`:

```bash
git clone https://github.com/LLM-Solution/PyLLMSol.git
cd PyLLMSol
pip install .
```

You can also install the dependencies directly:

```bash
pip install -r requirements.txt
```

## Features

- **Training Management**: Handle datasets, manage training steps, and track losses.
- **Checkpointing**: Save and load checkpoints of models and data at regular intervals.
- **CLI Interface**: Interact with the model via command-line.
- **API Support**: Host the model as a REST API with Flask.
- **Prompt Management**: Handle prompts with truncation and formatting options.


## Structure

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
PyLLMSol requires Python 3.6 or later. Core dependencies include:

- `flask>=3.0.3`
- `torch>=2.5.0`
- `transformers>=4.45.2`
- `tqdm>=4.66.5`
- `llama-cpp-python>=0.3.1`

For a full list, see requirements.txt.

## Getting Started

1. Clone the repository and install dependencies.
2. Set up your model files and ensure you have the necessary model weights.
3. Use the CLI or API modules to interact with the model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Arthur Bernard - contact@llm-solutions.fr

___

For further information, refer to the documentation in the source files.

