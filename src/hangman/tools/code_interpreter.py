import os
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_core.tools import Tool
from pydantic.v1 import BaseModel, Field

from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution

# Load environment variables from .env.local at the project root
load_dotenv('.env.local')


# --- Pydantic Schema for Clear Tool Input ---

class CodeInterpreterInput(BaseModel):
    """Input schema for the Code Interpreter tool."""
    code: str = Field(description="The Python code to be executed in the sandboxed Jupyter environment.")


# --- Helper Function for Formatting Tool Output ---

def format_e2b_output_to_str(observation: Dict[str, Any]) -> str:
    """
    Formats the structured dictionary output from the E2B tool into a single,
    human-readable string to be passed back to the LLM.
    """
    output = ""

    if error := observation.get("error"):
        # If there's a runtime error, prioritize showing it
        output += f"--- Python Error ---\n"
        output += f"Error Type: {error.get('name', 'Unknown Error')}\n"
        output += f"Error Value: {error.get('value', 'No details')}\n"
        output += f"Traceback:\n{error.get('traceback', 'No traceback')}\n"
        # Stop here if there's an error, as other output is less relevant
        return output

    if stderr := observation.get("stderr"):
        output += f"--- Stderr ---\n{stderr}\n"

    if stdout := observation.get("stdout"):
        output += f"--- Stdout ---\n{stdout}\n"
    
    if results := observation.get("results"):
        # Instead of dumping raw data, notify the agent that rich artifacts were made
        output += f"\n--- Rich Output ---\n[INFO] {len(results)} artifact(s) were generated (e.g., plots, data visualisations).\n"

    # If there was no output at all, confirm successful execution
    if not output:
        return "[SUCCESS] Code executed successfully with no output."

    return output.strip()


# --- Main Tool Class ---

class E2BCodeInterpreterTool:
    """
    A tool that wraps the E2B Code Interpreter sandbox, providing a stateful,
    secure, and powerful environment for Python code execution.
    
    This class manages the lifecycle of the sandbox and exposes a LangChain-compatible
    tool interface.
    """
    _sandbox: Sandbox

    def __init__(self):
        """
        Initializes the E2B Code Interpreter.
        It checks for the API key and instantiates the E2B Sandbox, which starts
        provisioning a cloud sandbox instance.
        """
        if not os.getenv("E2B_API_KEY"):
            raise ValueError(
                "E2B_API_KEY not found in environment variables. "
                "Please get your key from https://e2b.dev and add it to your .env.local file."
            )
        print("Initializing E2B Sandbox... (this may take a moment)")
        # This call starts the provisioning of the cloud sandbox
        self._sandbox = Sandbox()
        print("E2B Sandbox is ready.")

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Runs the given code in the E2B sandbox and returns the structured result.
        
        Args:
            code: The Python code to execute.
        
        Returns:
            A dictionary containing stdout, stderr, errors, and rich media results.
        """
        print(f"\n--- EXECUTING CODE ---\n{code}\n----------------------")
        execution: Execution = self._sandbox.run_code(code) 
        
        error_info = None
        if execution.error:
            error_info = {
                "name": execution.error.name,
                "value": execution.error.value,
                "traceback": "\n".join(execution.error.traceback)
            }
            
        return {
            "stdout": "\n".join(log.line for log in execution.logs.stdout),
            "stderr": "\n".join(log.line for log in execution.logs.stderr),
            "error": error_info,
            "results": execution.results # This list contains rich outputs like plots
        }
    
    def as_langchain_tool(self) -> Tool:
        """
        Creates and returns a LangChain Tool instance from this interpreter.
        This is the object that will be passed to the agent.
        """
        return Tool(
            name="code_interpreter",
            description=(
                "Executes Python code in a stateful Jupyter notebook environment. "
                "Use this for data analysis, calculations, plotting, and solving complex problems. "
                "The environment has many libraries pre-installed (e.g., pandas, numpy, matplotlib, seaborn, scikit-learn). "
                "It can generate and display plots or other files. Network access is disabled."
            ),
            func=self.execute,
            args_schema=CodeInterpreterInput,
        )

    def close(self):
        """
        Terminates the E2B sandbox session. This should be called at the end of
        the agent's lifecycle to release cloud resources.
        """
        if hasattr(self, '_sandbox') and self._sandbox:
            print("Closing E2B Sandbox...")
            self._sandbox.kill()
            print("E2B Sandbox closed.")