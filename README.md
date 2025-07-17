# AI Agent with Working Memory & Reasoning Distillation
This project is a command-line conversational AI agent built using LangGraph. It features a private "working memory" that allows it to maintain context and evolve its understanding throughout a conversation, powered by a reasoning distillation loop.
The agent's backend is a local language model served with vLLM, providing an efficient, OpenAI-compatible API for inference.

## Key Features
 * 🧠 **Private Working Memory**: The agent maintains an internal state that is updated after each interaction.
 * 🤔 **Reasoning Distillation**: Uses a "diff and patch" model to reflect on conversations and decide what information to add or modify in its memory.
 * ⚙️ **Modular LLM Backend**: Easily connects to any model served via a vLLM server.
 * 💬 **Interactive CLI**: A simple and straightforward command-line interface for chatting with the agent.

## 🚀 Setup and Usage
Follow these steps to set up the environment and run the agent.
1. **Create a Conda Environment**
First, create and activate a new Conda environment from your project's root directory.
``` bash
conda create --prefix=./venv python=3.11
conda activate ./venv
```

2. **Install Dependencies**
Install all the required Python packages using pip.
```bash
pip install -r requirements.txt
```

3. **Serve the Language Model (Terminal 1)**
In your first terminal, start the vLLM server. This command will download the specified model and serve it at http://localhost:8000. Leave this terminal running.
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --trust-remote-code
```

4. Run the Agent (Terminal 2)
In a second terminal (with the venv still activated), run the agent script. You can now start chatting with the agent.
```bash
python agent.py
```

And start playing! Try this prompt:

```
Let's play Hangman! You be the host. Think of a secret word, but don't tell me what it is. I'll try to guess it, one letter at a time. Just show me the blank spaces for the word to start.
```