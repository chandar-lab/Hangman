"""
MemoryOS Agent for Hangman SCT Testing.

Uses the three-tier hierarchical memory system:
- Short-term: Recent QA pairs
- Mid-term: Consolidated segments with heat scoring
- Long-term: User profile and assistant knowledge
"""
import os
import yaml
from collections import deque
from typing import List, Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider

from memoryos import Memoryos

from hangman.prompts.memoryos_agent import MAIN_SYSTEM_PROMPT


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    last_question: str
    last_system_prompt: str
    retrieved_memories: str


class MemoryOSAgent(BaseAgent):
    """
    MemoryOS-based agent with hierarchical memory.
    
    Flow:
    1. Retrieve relevant memories from all tiers
    2. Generate response with memory-augmented prompt
    3. Store (query, response) pair via add_memory()
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        memoryos_config: Optional[Dict[str, Any]] = None,
        memoryos_config_path: Optional[str] = None,
        session_id: str = "default",
        m_recent: int = 10,
        short_term_capacity: int = 10,
        mid_term_heat_threshold: int = 5,
        k_retrieve: int = 7,
        branch_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ):
        """
        Args:
            llm_provider: Pre-initialized provider (for response generation)
            memoryos_config: (optional) dict for MemoryOS configuration
            memoryos_config_path: (optional) YAML path if you prefer a file
            session_id: Used to namespace memory storage per experiment
            m_recent: Sliding window size for conversation context
            short_term_capacity: Max items in short-term memory (default: 10)
            mid_term_heat_threshold: Heat threshold for mid-term consolidation (default: 5)
            k_retrieve: Number of memories to retrieve (default: 7)
            branch_id: (optional) For SCT forking - creates a branch-specific session_id
            parent_session_id: (optional) For SCT forking - parent session to copy memories from
        """
        # Handle branching
        if branch_id is not None:
            base_session = parent_session_id or session_id
            self.session_id = f"{base_session}__branch_{branch_id}"
            self.parent_session_id = base_session
            self.branch_id = branch_id
        else:
            self.session_id = session_id
            self.parent_session_id = None
            self.branch_id = None

        self.m_recent = int(m_recent)
        self.k_retrieve = int(k_retrieve)
        self._thread_id = f"memoryos_main__{self.session_id}"

        # Load config
        if memoryos_config is None and memoryos_config_path:
            with open(memoryos_config_path, "r") as f:
                memoryos_config = yaml.safe_load(f)
        
        self._memoryos_config = memoryos_config or {}
        self._memoryos_config_path = memoryos_config_path

        # Get API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        # Determine LLM model from config or use default
        llm_model = self._memoryos_config.get("llm_model", "openai/gpt-4o-mini")

        # Initialize MemoryOS with session-specific user/assistant IDs
        self.memo = Memoryos(
            user_id=f"{self.session_id}_user",
            assistant_id=f"{self.session_id}_assistant",
            openai_api_key=api_key,
            openai_base_url="https://openrouter.ai/api/v1",
            data_storage_path=self._memoryos_config.get(
                "data_storage_path", 
                f"./memoryos_data/{self.session_id}"
            ),
            llm_model=llm_model,
            short_term_capacity=short_term_capacity,
            mid_term_heat_threshold=mid_term_heat_threshold,
            retrieval_queue_capacity=k_retrieve,
            long_term_knowledge_capacity=100,
        )

        # Sliding window for composing LLM messages
        self._window: deque = deque([], maxlen=self.m_recent)

        # Store parameters for cloning
        self._short_term_capacity = short_term_capacity
        self._mid_term_heat_threshold = mid_term_heat_threshold

        super().__init__(llm_provider=llm_provider)
        self.reset()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.set_entry_point("agent")
        return workflow.compile(checkpointer=MemorySaver())

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        incoming: List[BaseMessage] = state.get("messages", [])

        # Update sliding window
        for msg in incoming:
            if isinstance(msg, (HumanMessage, AIMessage)):
                self._window.append(msg)

        # Extract latest user question
        question = ""
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break

        # Retrieve memories from all tiers
        memories_str = self._retrieve_all_memories(question)

        # Build system prompt with memories
        system_prompt = MAIN_SYSTEM_PROMPT.format(memories=memories_str)
        messages_for_model: List[BaseMessage] = [
            SystemMessage(content=system_prompt)
        ] + list(self._window)

        # Generate response
        response_obj = self.llm_provider.client.invoke(messages_for_model)
        try:
            parsed = self.llm_provider.parse_response(response_obj.content or "")
            final_text = parsed.get("response") or (response_obj.content or "")
            thinking = parsed.get("thinking", "")
        except Exception:
            final_text = response_obj.content or ""
            thinking = ""

        # Store interaction in MemoryOS
        if question and final_text:
            try:
                self.memo.add_memory(
                    user_input=question,
                    agent_response=final_text
                )
            except Exception:
                pass  # Don't crash if memory storage fails

        ai_msg = AIMessage(content=final_text)

        return {
            "messages": state.get("messages", []) + [ai_msg],
            "thinking": thinking,
            "last_question": question,
            "last_system_prompt": system_prompt,
            "retrieved_memories": memories_str,
        }

    def _retrieve_all_memories(self, query: str) -> str:
        """Retrieve and format memories from all MemoryOS tiers."""
        sections = []

        # Short-term (recent QA pairs)
        try:
            st = self.memo.short_term_memory.get_all()
            if st:
                lines = ["## Recent Interactions"]
                for qa in st[-5:]:  # Last 5 for context
                    u = qa.get("user_input", "")[:200]
                    a = qa.get("agent_response", "")[:200]
                    lines.append(f"- User: {u}\n  Assistant: {a}")
                sections.append("\n".join(lines))
        except Exception:
            pass

        # User profile (long-term)
        try:
            profile = self.memo.get_user_profile_summary()
            if profile:
                sections.append(f"## User Profile\n{profile}")
        except Exception:
            pass

        # Assistant knowledge (long-term)
        try:
            knowledge = self.memo.get_assistant_knowledge_summary()
            if knowledge:
                if isinstance(knowledge, list):
                    # Extract just the 'knowledge' text, not the embeddings
                    knowledge_texts = []
                    for k in knowledge:
                        if isinstance(k, dict):
                            knowledge_texts.append(k.get("knowledge", str(k)))
                        else:
                            knowledge_texts.append(str(k))
                    knowledge = "\n".join(f"- {kt}" for kt in knowledge_texts)
                sections.append(f"## Learned Knowledge\n{knowledge}")
        except Exception:
            pass

        return "\n\n".join(sections) if sections else "(no memories retrieved)"

    # --- BaseAgent interface ---

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
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
        snap = self.workflow.get_state({"configurable": {"thread_id": self._thread_id}})
        return snap.values if snap else {}

    def get_private_state(self) -> str:
        """Dump all MemoryOS memory tiers for logging."""
        sections = []

        # Short-term
        try:
            st = self.memo.short_term_memory.get_all()
            lines = ["[Short-Term Memory]"]
            for qa in st:
                u = qa.get("user_input", "")[:100]
                a = qa.get("agent_response", "")[:100]
                ts = qa.get("timestamp", "")
                lines.append(f"- Q: {u}... A: {a}... @ {ts}")
            sections.append("\n".join(lines) if st else "[Short-Term Memory] (empty)")
        except Exception:
            sections.append("[Short-Term Memory] (error)")

        # Mid-term sessions
        try:
            mt = self.memo.mid_term_memory.sessions
            if mt:
                lines = ["[Mid-Term Memory]"]
                for sid, session in mt.items():
                    heat = session.get("H_segment", 0)
                    pages = len(session.get("details", []))
                    lines.append(f"- Session {sid}: {pages} pages, heat={heat:.2f}")
                sections.append("\n".join(lines))
            else:
                sections.append("[Mid-Term Memory] (empty)")
        except Exception:
            sections.append("[Mid-Term Memory] (error)")

        # Long-term user profile
        try:
            profile = self.memo.get_user_profile_summary()
            sections.append(f"[User Profile]\n{profile or '(none)'}")
        except Exception:
            sections.append("[User Profile] (error)")

        # Long-term assistant knowledge
        try:
            knowledge = self.memo.get_assistant_knowledge_summary()
            if knowledge:
                if isinstance(knowledge, list):
                    # Extract just the 'knowledge' text, not the embeddings
                    knowledge_texts = []
                    for k in knowledge:
                        if isinstance(k, dict):
                            knowledge_texts.append(k.get("knowledge", str(k)))
                        else:
                            knowledge_texts.append(str(k))
                    knowledge = "\n".join(f"- {kt}" for kt in knowledge_texts)
                sections.append(f"[Assistant Knowledge]\n{knowledge}")
            else:
                sections.append("[Assistant Knowledge] (none)")
        except Exception:
            sections.append("[Assistant Knowledge] (error)")

        return "\n\n".join(sections)

    def reset(self) -> None:
        self._window.clear()
        empty: AgentState = AgentState(messages=[], thinking="")
        self.workflow.update_state(
            {"configurable": {"thread_id": self._thread_id}}, 
            empty
        )

    # --- SCT Forking Support ---

    def get_session_config(self) -> Dict[str, Any]:
        """Return config for creating branch agents."""
        return {
            "llm_provider": self.llm_provider,
            "memoryos_config": self._memoryos_config,
            "memoryos_config_path": self._memoryos_config_path,
            "session_id": self.parent_session_id or self.session_id,
            "m_recent": self.m_recent,
            "short_term_capacity": self._short_term_capacity,
            "mid_term_heat_threshold": self._mid_term_heat_threshold,
            "k_retrieve": self.k_retrieve,
        }

    def clone_memories_from(self, parent_memo: "Memoryos") -> None:
        """
        Clone memories from parent MemoryOS instance.
        
        For SCT forking: copies short-term and mid-term memories to branch.
        Long-term is typically empty during short trials.
        """
        try:
            # Clone short-term memories
            parent_st = parent_memo.short_term_memory.get_all()
            for qa in parent_st:
                self.memo.add_memory(
                    user_input=qa.get("user_input", ""),
                    agent_response=qa.get("agent_response", "")
                )
        except Exception as e:
            print(f"Warning: Failed to clone MemoryOS memories: {e}")

    def get_sliding_window_state(self) -> List[BaseMessage]:
        """Return sliding window for seeding branch agents."""
        return list(self._window)


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Load environment variables (including OPENROUTER_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("Make sure you have a .env file with OPENROUTER_API_KEY set")
        raise SystemExit(1)
    
    print(f"✅ Environment loaded")
    
    CONFIG_PATH = "config/config.yaml"
    MEMORYOS_CONFIG_PATH = "config/memoryos_config_gptoss_20b.yaml"

    print("Is config file readable:", os.access(CONFIG_PATH, os.R_OK))
    print("Is MemoryOS config file readable:", os.access(MEMORYOS_CONFIG_PATH, os.R_OK))

    # Load your OpenRouter-backed LLMProvider
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="gpt_oss_20b_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        raise SystemExit(1)

    # Initialize the MemoryOSAgent
    try:
        agent = MemoryOSAgent(
            llm_provider=main_llm,
            memoryos_config_path=MEMORYOS_CONFIG_PATH,
            session_id="memoryos_session_1",
            m_recent=10,
            short_term_capacity=10,
            mid_term_heat_threshold=5,
            k_retrieve=7,
        )
        print("🤖 MemoryOSAgent is ready. Type 'quit', 'exit', or 'q' to end.")
    except Exception as e:
        print(f"❌ Failed to initialize MemoryOSAgent: {e}")
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

        print("\n---MEMORYOS STORED MEMORIES---")
        print(agent.get_private_state())

        print("\n" + "=" * 50)

