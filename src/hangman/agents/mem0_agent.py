# Remember to run the singularity qdrant server:
# module load singularity 
# singularity exec   --env QDRANT__SERVICE__HTTP_PORT=6333   --env QDRANT__SERVICE__GRPC_PORT=6334   --env QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage   --bind $SCRATCH/qdrant_storage:/qdrant/storage   $SCRATCH/containers/qdrant.sif /qdrant/qdrant

import os
import yaml
from collections import deque
from typing import List, Any, Dict, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider

# Paper QA prompt (Appendix “Prompt Template for Results Generation (Mem0)”)
# You said this exists here:
from hangman.prompts.mem0_agent import MAIN_SYSTEM_PROMPT
import litellm

# Patch BOTH entry points to be safe (some code calls litellm.supports_function_calling,
# others import it from litellm.utils). We wrap the original if it exists.
try:
    _orig_main = getattr(litellm, "supports_function_calling", None)
except Exception:
    _orig_main = None

from litellm import utils as _lu
_orig_utils = getattr(_lu, "supports_function_calling", None)

WHITELIST = {
    "openrouter/qwen/qwen3-32b",
    "openrouter/qwen/qwen3-235b-a22b-thinking-2507",
}

def _patched(model: str) -> bool:
    if model in WHITELIST:
        return True
    # fall back to originals if present
    if _orig_main is not None and _orig_main is not _patched:
        try:
            return _orig_main(model)
        except Exception:
            pass
    if _orig_utils is not None and _orig_utils is not _patched:
        try:
            return _orig_utils(model)
        except Exception:
            pass
    return False

# apply to both places
litellm.supports_function_calling = _patched
_lu.supports_function_calling = _patched
# ---- now it's safe to import mem0 or anything else ----

from mem0 import Memory


# ------------------------------
# Agent State
# ------------------------------
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    # Audit/debug fields:
    last_question: str
    last_system_prompt: str
    retrieved_user_memories: List[str]


# ------------------------------
# Mem0Agent
# ------------------------------
class Mem0Agent(BaseAgent):
    """
    Paper-faithful Mem0 baseline:
      - Extraction+Update: mem0.Memory.add(..., infer=True) with a sliding window of m messages.
      - Generation: retrieve top-k memories per speaker for the question and
        fill the paper’s QA template (single user message).
      - No graph memory / no rerankers / no proxy features.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        mem0_config: Optional[Dict[str, Any]] = None,
        mem0_config_path: Optional[str] = None,
        session_id: str = "default",
        m_recent: int = 10,          # paper: m = 10
        s_neighbors: int = 10,       # paper: s = 10 (neighbor set during update; library default if not exposed)
        k_per_user: int = 10,        # how many memories to surface to the QA prompt per speaker
        branch_id: Optional[str] = None,  # For SCT forking: unique branch identifier
        parent_session_id: Optional[str] = None,  # For SCT forking: parent session to clone from
    ):
        """
        Args:
            llm_provider: your pre-initialized provider (OpenRouter compatible).
            mem0_config: (optional) dict for Memory.from_config
            mem0_config_path: (optional) YAML path if you prefer a file
            session_id: used to namespace per-experiment memory keys
            m_recent: sliding window size passed to mem0.add(...) each turn
            s_neighbors: documented knob; OSS may use internal default if not exposed
            k_per_user: number of retrieved memories per speaker for generation
            branch_id: (optional) For SCT forking - creates a branch-specific session_id
            parent_session_id: (optional) For SCT forking - parent session to copy memories from
        """
        # Handle branching: if branch_id provided, create a unique session
        if branch_id is not None:
            base_session = parent_session_id if parent_session_id else session_id
            self.session_id = f"{base_session}__branch_{branch_id}"
            self.parent_session_id = base_session
            self.branch_id = branch_id
        else:
            self.session_id = session_id
            self.parent_session_id = None
            self.branch_id = None
        
        self.m_recent = int(m_recent)
        self.s_neighbors = int(s_neighbors)
        self.k_per_user = int(k_per_user)
        # --- in __init__ ---
        self._thread_id = f"mem0_main__{self.session_id}"

        # Build Mem0 client (OSS)
        if mem0_config is None and mem0_config_path is not None:
            with open(mem0_config_path, "r") as f:
                mem0_config = yaml.safe_load(f)
        self.mem0 = Memory.from_config(mem0_config or {})
        
        # Store config for cloning
        self._mem0_config = mem0_config
        self._mem0_config_path = mem0_config_path

        # Two "speaker memory buckets" (paper uses two-speaker memories)
        # We namespace by session_id so multiple runs don't collide.
        self.user_bucket = f"{self.session_id}__user"
        # self.assistant_bucket = f"{self.session_id}__assistant"

        # Sliding window of recent public messages (paper m=10)
        self._window: deque = deque([], maxlen=self.m_recent)

        # Parent init (sets self.llm_provider and builds workflow)
        super().__init__(llm_provider=llm_provider)

        # Reset state store
        self.reset()

    # ------------- LangGraph wiring -------------

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.set_entry_point("agent")
        return workflow.compile(checkpointer=MemorySaver())

    def _to_mem0_messages(self, msgs: List[BaseMessage]) -> List[dict]:
        """
        Convert LangChain messages -> Mem0 OSS expected format.
        Keeps only user/assistant/system roles; drops tools, function calls, etc.
        """
        out: List[dict] = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                out.append({"role": "assistant", "content": str(m.content)})
            elif isinstance(m, SystemMessage):
                out.append({"role": "system", "content": str(m.content)})
            # ignore ToolMessage and others for memory extraction
        return out

    # ------------- Core logic -------------

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        One-step node:
        1) Update Mem0 memories with last m messages (extraction+update).
        2) Retrieve relevant memories for the latest user turn.
        3) Compose OpenAI-style messages as:
            [ System(MAIN_SYSTEM_PROMPT with memories), last m conversation messages ]
        4) Call LLM and return AIMessage + audit fields.
        """
        incoming: List[BaseMessage] = state.get("messages", [])

        # Append new public messages into the sliding window (Human + AI only)
        for msg in incoming:
            if isinstance(msg, (HumanMessage, AIMessage)):
                self._window.append(msg)

        # --- 1) Extraction + Update (paper: pass last m messages; summary S if OSS supports it) ---
        window_list = list(self._window)                      # last m messages (Human + AI)
        mem0_msgs = self._to_mem0_messages(window_list)       # convert LC -> dict(role, content)
        if mem0_msgs:
            self.mem0.add(mem0_msgs, user_id=self.user_bucket, infer=True)

        # --- 2) Retrieval query = latest Human message content ---
        question = ""
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break

        try:
            res = self.mem0.search(query=question or "", user_id=self.user_bucket, limit=self.k_per_user) or {}
            results = res.get("results", []) or []
            user_mems = [r.get("memory", "") for r in results if r.get("memory")]
        except Exception:
            user_mems = []

        memories_str = "\n".join(user_mems) if user_mems else "(none)"

        # --- 3) Build model messages in README style ---
        # System contains the memories; then we append the last m turns (Human + AI)
        system_prompt = MAIN_SYSTEM_PROMPT.format(user_memories=memories_str)
        messages_for_model: List[BaseMessage] = [SystemMessage(content=system_prompt)] + window_list

        # --- 4) Generate with provider (temperature 0 recommended for exactness) ---
        response_obj = self.llm_provider.client.invoke(messages_for_model)
        try:
            parsed = self.llm_provider.parse_response(response_obj.content or "")
            final_text = parsed.get("response") or (response_obj.content or "")
            thinking = parsed.get("thinking", "")
        except Exception:
            final_text = response_obj.content or ""
            thinking = ""

        ai_msg = AIMessage(content=final_text)

        return {
            "messages": state.get("messages", []) + [ai_msg],
            "thinking": thinking,
            "last_question": question,
            "last_system_prompt": system_prompt,   # for audit
            "retrieved_user_memories": user_mems,
        }

    # ------------- Helpers -------------

    def _extract_question(self, incoming: List[BaseMessage]) -> str:
        """
        Paper QA step expects a question string. We take the latest HumanMessage content.
        If none, we fall back to empty.
        """
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def _retrieve_memories(self, bucket_user_id: str, query: str, limit: int) -> List[str]:
        try:
            res = self.mem0.search(query=query or "", user_id=bucket_user_id, limit=limit) or {}
            results = res.get("results", []) or []
            return [r.get("memory", "") for r in results if r.get("memory")]
        except Exception:
            return []

    # ------------- BaseAgent interface -------------
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """
        One-turn run with a persistent thread_id (required by MemorySaver).
        """
        cfg = {"configurable": {"thread_id": self._thread_id}}
        final_state = self.workflow.invoke({"messages": messages}, config=cfg)

        final_response = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                final_response = m.content
                break
        return {"response": final_response, "thinking": final_state.get("thinking", "")}

    # --- replace get_state() to read the same thread_id ---
    def get_state(self) -> Dict[str, Any]:
        snap = self.workflow.get_state({"configurable": {"thread_id": self._thread_id}})
        return snap.values if snap else {}
        
    def get_private_state(self) -> str:
        """
        Return the current contents of Mem0 for this agent/session.

        We fetch all memories for both buckets:
        - user bucket:     self.user_bucket  (e.g., "exp1__user")
        - assistant bucket:self.assistant_bucket

        The shape returned by mem0.get_all(...) is typically:
        {"results": [{"id": "...", "memory": "...", "metadata": {...}, ...}, ...]}
        """
        def _dump_bucket(bucket_label: str, bucket_user_id: str) -> str:
            try:
                res = self.mem0.get_all(user_id=bucket_user_id) or {}
                items = res.get("results", []) or []
                if not items:
                    return f"[{bucket_label} | {bucket_user_id}] (empty)"
                lines = [f"[{bucket_label} | {bucket_user_id}]"]
                for it in items:
                    mid = it.get("id") or it.get("_id") or "?"
                    text = it.get("memory") or it.get("text") or ""
                    ts = it.get("created_at") or it.get("timestamp") or it.get("time")
                    if ts:
                        # lines.append(f"- ({mid}) {text}  @ {ts}")
                        lines.append(f"- {text}  @ {ts}")
                    else:
                        # lines.append(f"- ({mid}) {text}")
                        lines.append(f"- {text}")
                return "\n".join(lines)
            except Exception as e:
                return f"[{bucket_label} | {bucket_user_id}] <error: {e}>"

        sections = [
            _dump_bucket("user", self.user_bucket),
            # _dump_bucket("assistant", self.assistant_bucket),
        ]
        return "\n\n".join(sections)

    def reset(self) -> None:
        self._window.clear()
        empty: AgentState = AgentState(messages=[], thinking="")
        self.workflow.update_state({"configurable": {"thread_id": self._thread_id}}, empty)

    # ------------- SCT Forking Support -------------
    
    def clone_memories_from(self, source_bucket: str) -> None:
        """
        Clone all memories from source_bucket to this agent's user_bucket.
        Used for SCT forking to ensure each branch starts with the same pre-fork memories.
        
        Args:
            source_bucket: The user_bucket identifier to copy from (e.g., "session__user")
        """
        try:
            # Retrieve all memories from source bucket
            res = self.mem0.get_all(user_id=source_bucket) or {}
            items = res.get("results", []) or []
            
            if not items:
                return  # No memories to clone
            
            # Extract and re-add each memory to the new bucket
            # Note: This will re-embed and store them under the new user_id
            for item in items:
                memory_text = item.get("memory") or item.get("text")
                if memory_text:
                    # Add as a single memory message (simplest approach)
                    # The memory will be re-embedded and stored in the new bucket
                    self.mem0.add(
                        messages=[{"role": "user", "content": memory_text}],
                        user_id=self.user_bucket,
                        infer=False  # Don't infer new memories, just store the existing one
                    )
        except Exception as e:
            # Log error but don't crash - branch can continue with empty memories
            print(f"Warning: Failed to clone memories from {source_bucket}: {e}")
    
    def get_session_config(self) -> Dict[str, Any]:
        """
        Return configuration needed to create a branch agent.
        Used by engine_sct.py to instantiate branch agents with the same settings.
        
        Returns:
            Dict with llm_provider, mem0_config, session params, etc.
        """
        return {
            "llm_provider": self.llm_provider,
            "mem0_config": self._mem0_config,
            "mem0_config_path": self._mem0_config_path,
            "session_id": self.parent_session_id or self.session_id,  # Use base session
            "m_recent": self.m_recent,
            "s_neighbors": self.s_neighbors,
            "k_per_user": self.k_per_user,
        }
    
    def get_sliding_window_state(self) -> List[BaseMessage]:
        """
        Return the current sliding window messages.
        Used by engine_sct.py to seed branch agents with pre-fork conversation state.
        
        Returns:
            List of BaseMessage objects in the sliding window
        """
        return list(self._window)

# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config/config.yaml"          # LLM + Mem0/Qdrant config
    MEM0_CONFIG_PATH = "config/mem0_config_gptoss_20b.yaml"

    print("Is file readable: ", os.access(CONFIG_PATH, os.R_OK))
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Load your OpenRouter-backed LLMProvider
    try:
        # e.g., "qwen3_235b_openrouter" or "gpt_oss_20b_openrouter" as you use in ReActMem
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="gpt_oss_20b_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        raise SystemExit(1)

    # Initialize the Mem0Agent
    agent = Mem0Agent(
        llm_provider=main_llm,
        mem0_config_path=MEM0_CONFIG_PATH,
        session_id="mem0_session_1",
        m_recent=10,       # paper: m=10
        s_neighbors=10,    # paper: s=10 (used internally by update; OSS may use default)
        k_per_user=10,     # retrieved memories per speaker for QA prompt
    )
    print("🤖 Mem0Agent is ready. Type 'quit', 'exit', or 'q' to end.")

    # Interactive loop (same shape as ReActMem)
    messages = []
    while True:
        try:
            user_input = input("User > ")
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending session.")
            break

        messages.append(HumanMessage(content=user_input))

        output = agent.invoke(messages)

        messages.append(AIMessage(content=output["response"]))

        print("\n---ANSWER---")
        print(f"AI: {output['response']}")

        # Print thinking trace if available
        if "thinking" in output and output["thinking"]:
            print("\n---THINKING TRACE---")
            print(output["thinking"])

        print("\n---MEM0 STORED MEMORIES---")
        print(agent.get_private_state())

        print("\n" + "=" * 50 + "\n")

