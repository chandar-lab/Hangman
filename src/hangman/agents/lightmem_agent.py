import os
import yaml
import uuid
import copy
from collections import deque
from typing import List, Any, Dict, Optional
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.prompts.lightmem_agent import MAIN_SYSTEM_PROMPT

# LightMem memory system
from lightmem.memory.lightmem import LightMemory


# ------------------------------
# Agent State
# ------------------------------
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    last_question: str
    last_system_prompt: str
    retrieved_memories: str


# ------------------------------
# LightMemAgent
# ------------------------------
class LightMemAgent(BaseAgent):
    """
    LightMem-based agent with:
    - Online memory extraction: add_memory() per turn with compression + segmentation
    - Retrieval: semantic search for relevant memories
    - Generation: System(memories) + sliding window
    - No offline updates during trials (embedding-only, no SQL)
    
    Memory lifecycle:
    1. Pre-compression (LLMLingua-2)
    2. Topic segmentation
    3. Metadata extraction with LLM
    4. Vector storage in Qdrant
    5. Semantic retrieval
    
    Environment Setup:
    - The __init__ automatically sets up OPENAI_API_KEY and OPENAI_BASE_URL from OPENROUTER_API_KEY
    - Just ensure OPENROUTER_API_KEY is in your environment (e.g., via .env file)
    - The class handles all environment variable swapping internally
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        lightmem_config: Optional[Dict[str, Any]] = None,
        lightmem_config_path: Optional[str] = None,
        session_id: str = "default",
        m_recent: int = 10,          # Sliding window size
        k_retrieve: int = 10,        # Retrieved memories for prompt
        branch_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ):
        """
        Args:
            llm_provider: Pre-initialized provider (for response generation)
            lightmem_config: (optional) dict for LightMemory.from_config
            lightmem_config_path: (optional) YAML path if you prefer a file
            session_id: Used to namespace Qdrant collections per experiment
            m_recent: Sliding window size for conversation context
            k_retrieve: Number of retrieved memories for generation
            branch_id: (optional) For SCT forking - creates a branch-specific session_id
            parent_session_id: (optional) For SCT forking - parent session to copy memories from
        """
        # CRITICAL: Setup OPENAI_* environment variables from OPENROUTER_* if needed
        # LightMem expects OPENAI_API_KEY and OPENAI_BASE_URL, but we use OpenRouter
        # This ensures the class works whether called from __main__ or run_sct_hangman.py
        # ALWAYS override any existing OPENAI_API_KEY with OPENROUTER_API_KEY
        if os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        
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
        self.k_retrieve = int(k_retrieve)
        self._thread_id = f"lightmem_main__{self.session_id}"

        # Load config from file if provided
        if lightmem_config is None and lightmem_config_path is not None:
            with open(lightmem_config_path, "r") as f:
                lightmem_config = yaml.safe_load(f)
        
        # Deep copy config to avoid mutation
        config_copy = copy.deepcopy(lightmem_config or {})
        
        # Inject API key from environment (LightMem expects it in config)
        if "memory_manager" in config_copy and "configs" in config_copy["memory_manager"]:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                config_copy["memory_manager"]["configs"]["api_key"] = api_key
        
        # Namespace Qdrant collection by session for isolation
        if "embedding_retriever" in config_copy:
            config_copy["embedding_retriever"]["configs"]["collection_name"] = \
                f"lightmem_{self.session_id}"
            config_copy["embedding_retriever"]["configs"]["path"] = \
                f"./qdrant_storage/lightmem_{self.session_id}"
        print(f"DEBUG [__init__]: session_id={self.session_id}, collection_name={config_copy['embedding_retriever']['configs']['collection_name']}")

        # CRITICAL: Temporarily remove OPENROUTER_* env vars during LightMem init
        # (LightMem gets confused by them and tries to use openrouter_base_url)
        # We'll restore them after so llm_provider can use them
        saved_openrouter_key = os.environ.pop("OPENROUTER_API_KEY", None)
        saved_openrouter_base = os.environ.pop("OPENROUTER_API_BASE", None)
        
        try:
            # Initialize LightMem with session-specific config (uses OPENAI_* env vars only)
            self.lightmem = LightMemory.from_config(config_copy)
        finally:
            # Restore OPENROUTER_* env vars for llm_provider to use later
            if saved_openrouter_key:
                os.environ["OPENROUTER_API_KEY"] = saved_openrouter_key
            if saved_openrouter_base:
                os.environ["OPENROUTER_API_BASE"] = saved_openrouter_base
        
        # Store config for cloning (use the modified config_copy with session-specific settings)
        self._lightmem_config = config_copy
        self._lightmem_config_path = lightmem_config_path

        # Sliding window for composing messages to the LLM
        self._window: deque = deque([], maxlen=self.m_recent)

        # Parent init (sets self.llm_provider and builds workflow)
        super().__init__(llm_provider=llm_provider)

        # Reset state store
        self.reset()

    # ------------- LangGraph wiring -------------

    def _build_workflow(self) -> StateGraph:
        """Build a minimal StateGraph for the agent."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.set_entry_point("agent")
        return workflow.compile(checkpointer=MemorySaver())

    def _to_lightmem_format(self, msgs: List[BaseMessage]) -> List[dict]:
        """
        Convert LangChain messages to LightMem format.
        LightMem requires: {"role": "user"|"assistant", "content": str, "time_stamp": str}
        
        We use ISO timestamps that LightMem's MessageNormalizer will parse and increment.
        """
        out = []
        # Use a base timestamp and let MessageNormalizer handle increments
        base_timestamp = datetime.now().strftime("%Y/%m/%d (%a) %H:%M")
        
        for msg in msgs:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue  # Skip system messages
            
            out.append({
                "role": role,
                "content": str(msg.content),
                "time_stamp": base_timestamp  # MessageNormalizer will auto-increment
            })
        
        return out

    # ------------- Core logic -------------

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        One-step node:
        1) Retrieve memories for the latest user question
        2) Generate response with memory-injected prompt
        3) Add turn to LightMem (async/post-response)
        """
        incoming: List[BaseMessage] = state.get("messages", [])

        # Update sliding window (Human + AI only)
        for msg in incoming:
            if isinstance(msg, (HumanMessage, AIMessage)):
                self._window.append(msg)

        # Extract latest user question
        question = ""
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break

        # --- 1) RETRIEVAL (pre-response) ---
        retrieved_str = ""
        try:
            retrieved_str = self.lightmem.retrieve(
                query=question or "",
                limit=self.k_retrieve,
                filters=None
            )
            # Returns formatted string: "timestamp weekday memory\n..."
        except Exception as e:
            retrieved_str = "(no memories retrieved)"
            print(f"Warning: LightMem retrieval failed: {e}")

        # --- 2) GENERATION with memory-injected prompt ---
        system_prompt = MAIN_SYSTEM_PROMPT.format(memories=retrieved_str)
        messages_for_model: List[BaseMessage] = [SystemMessage(content=system_prompt)] + list(self._window)

        # Call LLM provider
        response_obj = self.llm_provider.client.invoke(messages_for_model)
        try:
            parsed = self.llm_provider.parse_response(response_obj.content or "")
            final_text = parsed.get("response") or (response_obj.content or "")
            thinking = parsed.get("thinking", "")
        except Exception:
            final_text = response_obj.content or ""
            thinking = ""

        # --- 3) MEMORY ADDITION (post-response, no offline update) ---
        window_list = list(self._window)
        lightmem_msgs = self._to_lightmem_format(window_list)
        
        if lightmem_msgs:
            try:
                # CRITICAL: Setup environment for LightMem's OpenAI client during add_memory
                # 1. Save OPENROUTER_* values BEFORE removing them
                # 2. Set OPENAI_* from saved values
                # 3. Remove OPENROUTER_* (they confuse LightMem)
                saved_openrouter_key = os.getenv("OPENROUTER_API_KEY")
                saved_openrouter_base = os.getenv("OPENROUTER_API_BASE")
                
                # Set OPENAI_* for LightMem's metadata generation
                if saved_openrouter_key:
                    os.environ["OPENAI_API_KEY"] = saved_openrouter_key
                os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
                
                # Now remove OPENROUTER_* to avoid confusion
                os.environ.pop("OPENROUTER_API_KEY", None)
                os.environ.pop("OPENROUTER_API_BASE", None)
                
                try:
                    self.lightmem.add_memory(
                        messages=lightmem_msgs,
                        force_segment=True,   # Force immediate segmentation for interactive use
                        force_extract=True    # Force immediate extraction for interactive use
                    )
                    # NOTE: We do NOT call offline update methods:
                    # - construct_update_queue_all_entries()
                    # - offline_update_all_entries()
                finally:
                    # Restore OPENROUTER_* env vars for llm_provider to use later
                    if saved_openrouter_key:
                        os.environ["OPENROUTER_API_KEY"] = saved_openrouter_key
                    if saved_openrouter_base:
                        os.environ["OPENROUTER_API_BASE"] = saved_openrouter_base
            except Exception as e:
                print(f"Warning: LightMem add_memory failed: {e}")

        ai_msg = AIMessage(content=final_text)

        return {
            "messages": state.get("messages", []) + [ai_msg],
            "thinking": thinking,
            "last_question": question,
            "last_system_prompt": system_prompt,
            "retrieved_memories": retrieved_str,
        }

    # ------------- BaseAgent interface -------------

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """One-turn run with a persistent thread_id."""
        cfg = {"configurable": {"thread_id": self._thread_id}}
        final_state = self.workflow.invoke({"messages": messages}, config=cfg)

        final_response = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                final_response = m.content
                break

        return {
            "response": final_response,
            "thinking": final_state.get("thinking", "")
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current LangGraph state."""
        snap = self.workflow.get_state({"configurable": {"thread_id": self._thread_id}})
        return snap.values if snap else {}

    def get_private_state(self) -> str:
        """
        Return the retrieved memories from the last turn for logging.
        This shows what memories were surfaced to the agent.
        """
        state = self.get_state()
        retrieved = state.get("retrieved_memories", "")
        
        if retrieved and retrieved != "(no memories retrieved)":
            return f"[LightMem Retrieved Memories]\n{retrieved}"
        else:
            return "[LightMem Retrieved Memories]\n(none)"

    def reset(self) -> None:
        """Reset agent state and clear sliding window."""
        self._window.clear()
        empty: AgentState = AgentState(messages=[], thinking="")
        self.workflow.update_state(
            {"configurable": {"thread_id": self._thread_id}},
            empty
        )

    # ------------- SCT Forking Support -------------

    def clone_memories_from(self, parent_lightmem) -> None:
        """
        Clone all memories from parent agent's LightMemory instance to this branch.
        
        Used for SCT forking to ensure each branch starts with identical pre-fork memories.
        Copies vector embeddings and payloads directly from parent's in-memory Qdrant instance.
        
        Args:
            parent_lightmem: The parent agent's LightMemory instance to clone from
                           (or a session_id string for backward compatibility - will fail gracefully)
        """
        try:
            # Handle backward compatibility: if given a string, it's a session_id
            if isinstance(parent_lightmem, str):
                print(f"WARNING [clone]: Received session_id string instead of LightMemory instance")
                print(f"WARNING [clone]: Cannot clone memories without direct access to parent's LightMemory")
                return
            
            # Get all memories from parent's LightMemory instance directly
            print(f"DEBUG [clone]: Attempting to retrieve all memories from parent agent")
            source_memories = parent_lightmem.embedding_retriever.get_all()
            print(f"DEBUG [clone]: Retrieved {len(source_memories) if source_memories else 0} memories from parent")
            
            if not source_memories:
                print(f"WARNING [clone]: No memories found in parent's LightMemory instance")
                return  # No memories to clone
            
            print(f"DEBUG [clone]: Starting to clone {len(source_memories)} memories to branch {self.session_id}")
            # Clone each memory to this branch with new UUIDs
            cloned_count = 0
            for entry in source_memories:
                vector = entry.get("vector")
                payload = entry.get("payload")
                
                if vector and payload:
                    # Insert with fresh UUID (Qdrant requirement)
                    self.lightmem.embedding_retriever.insert(
                        vectors=[vector],
                        payloads=[payload],
                        ids=[str(uuid.uuid4())]
                    )
                    cloned_count += 1
                else:
                    print(f"DEBUG [clone]: Skipped entry with missing vector or payload")
            
            print(f"✓ Cloned {cloned_count}/{len(source_memories)} memories to branch {self.session_id}")
            
        except Exception as e:
            # Don't crash if cloning fails - branch can continue with empty memories
            print(f"Warning: Failed to clone LightMem memories: {e}")
            import traceback
            traceback.print_exc()

    def get_session_config(self) -> Dict[str, Any]:
        """
        Return configuration needed to create a branch agent.
        Used by engine_sct.py to instantiate branch agents with the same settings.
        
        Returns:
            Dict with llm_provider, lightmem_config, session params, etc.
        """
        return {
            "llm_provider": self.llm_provider,
            "lightmem_config": self._lightmem_config,
            "lightmem_config_path": self._lightmem_config_path,
            "session_id": self.parent_session_id or self.session_id,  # Use base session
            "m_recent": self.m_recent,
            "k_retrieve": self.k_retrieve,
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
    from dotenv import load_dotenv
    
    # Load environment variables (including OPENROUTER_API_KEY)
    # The __init__ method will handle the OPENROUTER → OPENAI key swap
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("Make sure you have a .env file with OPENROUTER_API_KEY set")
        raise SystemExit(1)
    
    print(f"✅ Environment loaded")
    
    CONFIG_PATH = "config/config.yaml"  # LLM provider config
    LIGHTMEM_CONFIG_PATH = "config/lightmem_config_gptoss_20b.yaml"  # LightMem-specific config

    print("Is config file readable:", os.access(CONFIG_PATH, os.R_OK))
    print("Is LightMem config file readable:", os.access(LIGHTMEM_CONFIG_PATH, os.R_OK))

    # Load your OpenRouter-backed LLMProvider
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="gpt_oss_20b_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        raise SystemExit(1)

    # Initialize the LightMemAgent
    # The __init__ will automatically handle OPENROUTER → OPENAI environment variable setup
    try:
        agent = LightMemAgent(
            llm_provider=main_llm,
            lightmem_config_path=LIGHTMEM_CONFIG_PATH,
            session_id="lightmem_session_1",
            m_recent=10,
            k_retrieve=10,
        )
        print("🤖 LightMemAgent is ready. Type 'quit', 'exit', or 'q' to end.")
    except Exception as e:
        print(f"❌ Failed to initialize LightMemAgent: {e}")
        raise SystemExit(1)

    # Interactive loop
    messages = []
    while True:
        try:
            user_input = input("\nUser > ")
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

        print("\n---LIGHTMEM RETRIEVED MEMORIES---")
        print(agent.get_private_state())

        print("\n" + "=" * 50)

