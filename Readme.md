# Questions AI

An AI project for processing and training models on question datasets.

## Setup Instructions

### Prerequisites

This project requires Python 3.12 or higher. Check your Python version:

```bash
python --version
```

### 1. Create Virtual Environment

Create a virtual environment using Python 3.12:

```bash
# Create virtual environment
py -3.12 -m venv venv312

# Activate the environment
# Windows
venv312\Scripts\activate

# Linux/macOS
source venv312/bin/activate
```

### 2. Install Dependencies

After activating the virtual environment, install the required packages:

```bash
pip install -r requirement.txt
```

Note: The requirements.txt has been updated to support Python 3.12 with the latest compatible package versions including:
- numpy 2.1.2 (latest stable version)
- scipy 1.14.1 (latest stable version)
- torch 2.5.1+cu121 (with CUDA 12.1 support)

### 3. Hugging Face Authentication

To access Hugging Face models and datasets, you need to authenticate with your Hugging Face token.

#### Option 1: Using Hugging Face CLI (Recommended)

1. Install the Hugging Face CLI (already included in requirements):
```bash
pip install huggingface_hub
```

2. Login with your Hugging Face token:
```bash
huggingface-cli login
```

3. Enter your token when prompted. You can get your token from:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose appropriate permissions (Read/Write)
   - Copy the generated token

4. The token will be automatically saved to `~/.cache/huggingface/token` for future use.

#### Option 2: Environment Variable

Alternatively, you can set an environment variable:

```bash
# Windows (PowerShell)
$env:HUGGINGFACE_HUB_TOKEN="your_token_here"

# Windows (Command Prompt)
set HUGGINGFACE_HUB_TOKEN=your_token_here

# Linux/macOS
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

#### Option 3: Using .env file (Development)

Create a `.env` file in the project root:
```env
HUGGINGFACE_HUB_TOKEN=your_token_here
```

⚠️ **Security Note**: Never commit your token to version control. Add `.env` to `.gitignore`.

### 3. Verify Installation

Test your setup by running:
```python
from huggingface_hub import whoami
print(whoami())
```

## Usage

[Add your usage instructions here]

## Project Structure

```
questions-ai/
├── cleanData/           # Data cleaning utilities
├── data/               # Raw datasets
├── cleaned_data/       # Processed datasets
├── requirement.txt     # Python dependencies
└── README.md          # This file
```
