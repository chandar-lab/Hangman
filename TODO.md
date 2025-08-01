# Project TODO & Architecture Guide
This document outlines the software architecture and a prioritized development plan for the "LLMs Can't Play Hangman" experimental framework. The goal is to create a modular, extensible codebase in Python using LangChain and LangGraph to test various agents' abilities to maintain a private state.
## 🏛️ Project Architecture
The architecture is designed to be modular, separating the core logic of the experiment from the specific implementations of agents, games, and players.
* **Providers**: A centralized module (`src/providers/`) for instantiating and configuring LLM clients (e.g., OpenRouter API, `vllm` local models). This ensures consistent model access across the project.
* **Engine**: The `GameLoopController` that orchestrates the entire interaction. It initializes a game, manages the turn-by-turn conversation between an agent and a player, and logs the results.
* **Agents**: The "brains" being tested. All agents inherit from a `BaseAgent` class, ensuring they can be seamlessly swapped out by the engine. This includes simple baselines and your `PrivateMemoryAgent` built with `LangGraph`.
* **Games**: Defines the rules, prompts, and win/loss conditions for each experimental task (Hangman, Zendo, etc.).
* **Players**: The LLM-powered bots that interact with the agents. This module contains the logic for both Cooperative and Adversarial playing styles.
* **Evaluation**: The `JudgeLLM` component responsible for analyzing the saved game transcripts and scoring the agents based on the defined multi-faceted rubric.

## 📁 Directory Structure
``` bash
/hangman/
|
|-- 📄 pyproject.toml         # Poetry's main config and dependency file
|-- 📄 poetry.lock            # Poetry's file for deterministic installs
|-- 📄 README.md              # Project documentation
|-- 📄 config.yaml            # For experiment configs (model names, runs, etc.)
|-- 📄 run_experiment.py      # Main script to launch experiments
|-- 📄 .env                   # For storing API keys (should be in .gitignore)
|-- 📄 .gitignore             # To ignore files like .env, .venv, __pycache__
|
|-- src
|   └── hangman/
|       |-- providers/ # Handles LLM provider logic
|       | |-- llm_provider.py
|       |-- engine.py # GameLoopController orchestrator
|       |-- agents/
|       | |-- base_agent.py
|       | |-- stateless_agent.py
|       | |-- react_agent.py
|       | |-- cogniact.py # Your agent using LangGraph
|       |-- games/
|       | |-- base_game.py
|       | |-- hangman.py
|       | |-- zendo.py
|       |-- players/
|       | |-- llm_player.py
|       |-- evaluation/
|       | |-- judge.py
|
|-- notebooks/               # For analysis and exploration
|   └── analyze_results.ipynb
|
|-- results/                 # To store raw JSON logs from experiments
|   |-- hangman/
|   |-- zendo/
|   └── .gitkeep
|
|-- tests/                   # For unit tests
|   └── .gitkeep
```

## ✅ Development Plan & TODO List
This list is prioritized to build the project from the ground up, ensuring a testable foundation at each stage.

### Priority 1: Setup & Foundations 🏗️
* ✅ **Project Scaffolding**: Create the directory structure and initialize a Git repository.
* ✅ **Environment Setup**: Set up a Python virtual environment (`.venv`), populate requirements.txt (with `langchain`, `langgraph`, `python-dotenv`, etc.), and create the `.env` file for API keys.
* ✅ **Configuration**: Implement the `config.yaml` to manage model names, API endpoints, and experiment parameters like the number of runs.
* ✅ **LLM Provider**: Create the `src/providers/llm_provider.py` to have a centralized factory function (`get_llm()`) that returns initialized `LangChain` model objects.

### Priority 2: Build a Single End-to-End Run 🔬
* ✅ **Define Interfaces**: Implement the abstract base classes: `BaseAgent`, `BaseGame`, and `BasePlayer`.
* ✅ Implement Core Components:
    * Create the `CogniAct` as the first simple baseline.
    * Create the `HangmanGame` class with its prompts and termination logic.
    * Create the `LLMPlayer` with only the Cooperative mode implemented initially.
* ✅ **Build the Engine**: Implement the `GameLoopController` in `src/engine.py`. The goal is to be able to run a single game of Hangman with the stateless agent and have it save a complete JSON log to the `results/` directory.

### Priority 3: Implement Advanced Logic & Evaluation 🧠
* **Develop Baselines**: Implement the more complex baseline agents, like the `ReActAgent`.
* **Build Your Agent**: Implement the `PrivateMemoryAgent` using `LangGraph`. This will be a major task, focusing on correctly defining the state, nodes, and edges.
* **Create the Judge**: Implement the `JudgeLLM` in `src/evaluation/judge.py`. It should be able to load a JSON log file and return a structured evaluation based on your 4-point rubric.

### Priority 4: Scale & Finalize Experiments 🚀
* **Expand Content**: Implement the remaining games (Zendo, Medical Diagnosis Simulator, etc.) and the Adversarial player mode.
* **Automate Experimentation**: Write the main logic in `run_experiment.py`. This script should read the config, loop through all combinations of (agent, game, player mode), and run the required number of trials, saving all results. Consider adding parallel processing to speed up API calls.
* **Analyze & Visualize**: Use the `analyze_results.ipynb` notebook to load the data from the `results/` folder, perform statistical analysis, and generate the plots and tables for your paper.
* **Code Refinement**: Add `docstrings`, type hints, and unit tests to improve code quality and reproducibility.

### Other
* Change messages -> messages
* add get_private_state method to all agents