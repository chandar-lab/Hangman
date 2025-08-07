import os
import json
import argparse
import asyncio
import sys
from typing import Dict, Any

# --- Third-Party Imports ---
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from langchain_core.messages import HumanMessage

# --- Hangman Project Imports ---
# Assumes 'hangman' is installed in the venv (e.g., `poetry install`)
try:
    from hangman.agents import create_agent
    from hangman.providers.llmprovider import LLMProvider, load_llm_provider
except ImportError:
    print("❌ Error: Could not import from 'hangman'.")
    print("Please ensure you have run 'poetry install' and are running this script from the project root.")
    sys.exit(1)

# --- Global dictionary to hold LLM providers ---
# We load these once in main() to avoid reloading models for every question.
LLM_PROVIDERS: Dict[str, LLMProvider] = {}

def format_question_text(question: Dict[str, Any]) -> str:
    """
    Formats the question from the HLE dataset into a single string.
    NOTE: This version is text-only. The image URL is ignored as the current
    agent framework does not support multi-modal inputs.
    """
    question_text = question['question']
    if question.get('image'):
        # For now, we just acknowledge that an image was part of the prompt.
        # A future implementation could handle this more gracefully.
        question_text += "\n[Note: An image was provided with this question, but is not displayed.]"
    return question_text

def run_inference_sync(question: Dict[str, Any], agent_type: str) -> str:
    """
    This is the synchronous, blocking function that runs the agent.
    It will be executed in a separate thread by asyncio.to_thread.
    """
    try:
        # 1. Instantiate a FRESH agent for each question to ensure no state leakage.
        agent = create_agent(
            agent_name=agent_type,
            main_llm_provider=LLM_PROVIDERS['main'],
            distillation_llm_provider=LLM_PROVIDERS.get('distill')
        )
        agent.reset()  # Ensure the agent starts with a clean state

        # 2. Format the question and invoke the agent
        prompt = format_question_text(question)
        messages = [HumanMessage(content=prompt)]
        
        # This is the primary blocking call (CPU/GPU bound)
        output = agent.invoke(messages)

        # 3. Return the final response string
        return output['response']

    except Exception as e:
        print(f"Error during agent invocation for question {question.get('id', 'N/A')}: {e}", file=sys.stderr)
        return None  # Return None on failure

async def attempt_question(question: Dict[str, Any], agent_type: str):
    """
    Asynchronous wrapper that runs the synchronous agent inference in a thread.
    """
    response_content = await asyncio.to_thread(run_inference_sync, question, agent_type)
    
    if response_content is None:
        return None # Propagate the failure

    # The original script returned usage tokens. We return an empty dict
    # as our agent framework does not expose this easily.
    usage_tokens = {} 
    
    return question["id"], response_content, usage_tokens

async def attempt_all(questions: list, agent_type: str, num_workers: int):
    """
    Manages the asynchronous execution of all questions using a semaphore
    to limit concurrency.
    """
    semaphore = asyncio.Semaphore(num_workers)

    async def bound_func(question):
        async with semaphore:
            return await attempt_question(question, agent_type)
            
    tasks = [bound_func(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating Agent: {agent_type}")
    return results

def main(args):
    # --- 1. Argument Validation and Setup ---
    if not args.output_file:
        # Create a default output file path if not provided
        os.makedirs("results/hle", exist_ok=True)
        args.output_file = f"results/hle/hle_{args.agent_type}.json"
        print(f"Warning: --output-file not specified. Defaulting to {args.output_file}")

    # --- 2. Load LLM Providers Once ---
    print("--- ⚙️  Initializing LLM Providers ---")
    try:
        LLM_PROVIDERS['main'] = load_llm_provider(args.config_path, provider_name=args.main_llm_name)
        if args.distill_llm_name:
            LLM_PROVIDERS['distill'] = load_llm_provider(args.config_path, provider_name=args.distill_llm_name)
        print("✅ LLM Providers initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Providers from '{args.config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Load and Prepare Dataset ---
    print(f"--- 📚 Loading dataset '{args.dataset}' ---")
    dataset = load_dataset(args.dataset, split="test").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    
    if args.max_samples:
        print(f"Limiting to first {args.max_samples} samples.")
        questions = questions[:args.max_samples]
    
    # --- 4. Load Existing Predictions to Avoid Rerunning ---
    if os.path.exists(args.output_file):
        print(f"Found existing results file: {args.output_file}. Loading...")
        with open(args.output_file, "r") as f:
            predictions = json.load(f)
        
        completed_ids = set(predictions.keys())
        original_count = len(questions)
        questions = [q for q in questions if q["id"] not in completed_ids]
        print(f"Resuming. Found {len(completed_ids)} completed questions. {len(questions)} remaining.")
    else:
        predictions = {}

    if not questions:
        print("✅ No new questions to process. All done!")
        return

    # --- 5. Run the Evaluation ---
    results = asyncio.run(attempt_all(questions, args.agent_type, args.num_workers))

    # --- 6. Process and Save Results ---
    failures = 0
    for result in results:
        if result is None:
            failures += 1
            continue
        
        unique_id, response, usage = result
        predictions[unique_id] = {
            "agent_type": args.agent_type,
            "response": response,
            "usage": usage
        }

    if failures > 0:
        print(f"⚠️  Warning: {failures} out of {len(results)} questions failed during processing.")

    print(f"--- 💾 Saving {len(predictions)} predictions to {args.output_file} ---")
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print("--- ✨ Evaluation run complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangman Agents on the Humanity's Last Exam (HLE) benchmark.")
    
    # --- Key Arguments for Agent and Data ---
    parser.add_argument("--dataset", type=str, default="cais/hle", help="Hugging Face dataset to use for evaluation.")
    parser.add_argument("--agent-type", type=str, required=True, help="The class name of the agent to test (e.g., 'ReActAgent', 'ReaDisUpdActAgent').")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save the JSON predictions. Defaults to 'results/hle/hle_{agent_type}.json'.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation to the first N samples for quick testing.")
    
    # --- Concurrency and System Config ---
    parser.add_argument("--num-workers", type=int, default=4, help="Number of concurrent agent instances to run. Tune based on your machine's CPU/GPU resources.")
    parser.add_argument("--config-path", type=str, default="config.yaml", help="Path to the main project configuration file.")
    
    # --- LLM Provider Config ---
    parser.add_argument("--main-llm-name", type=str, default="qwen3_14b_local", help="Name of the main LLM provider in config.yaml.")
    parser.add_argument("--distill-llm-name", type=str, default="qwen3_14b_local", help="Name of the distillation LLM provider in config.yaml (if required by the agent).")

    args = parser.parse_args()
    main(args)