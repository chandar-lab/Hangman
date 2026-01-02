# MemoryOS Integration Plan

## Executive Summary

This document outlines the integration of [MemoryOS](https://github.com/BAI-LAB/MemoryOS) as a memory-based agent in the Hangman SCT (Self-Consistency Test) benchmark. MemoryOS is a hierarchical memory system inspired by operating system memory management, featuring short-term, mid-term, and long-term memory tiers.

---

## 1. Understanding MemoryOS Architecture

### 1.1 Core Concepts

MemoryOS uses a **three-tier hierarchical memory architecture**:

| Tier | Purpose | Capacity | Persistence |
|------|---------|----------|-------------|
| **Short-term** | Recent QA pairs (sliding window) | Configurable (default: 10) | Session |
| **Mid-term** | Consolidated interaction segments with "heat" scoring | Session-based | Session |
| **Long-term** | User profile + assistant knowledge base | Persistent | Across sessions |

### 1.2 Memory Flow

```
User Query → Retrieval (all tiers) → LLM Response → add_memory() → Short-term
                                                            ↓
                                         (when capacity reached)
                                                            ↓
                                                       Mid-term
                                                            ↓
                                         (when heat threshold exceeded)
                                                            ↓
                                                       Long-term
```

### 1.3 Key API Methods

From the notebook exploration (`notebooks/memoryos.ipynb`):

```python
from memoryos import Memoryos

memo = Memoryos(
    user_id="user_123",
    assistant_id="assistant_123",
    openai_api_key=api_key,
    openai_base_url="https://openrouter.ai/api/v1",
    data_storage_path="./memoryos_data",
    llm_model="openai/gpt-4o-mini",
    short_term_capacity=10,
    mid_term_heat_threshold=5,
    retrieval_queue_capacity=7,
    long_term_knowledge_capacity=100,
)

# Main interaction pattern
response = memo.get_response(query=user_input)  # Retrieves + generates + stores

# Manual add (alternative)
memo.add_memory(user_input=text, agent_response=response)

# Inspection methods
memo.get_user_profile_summary()        # Long-term user profile
memo.get_assistant_knowledge_summary() # Long-term assistant knowledge
memo.short_term_memory.get_all()       # Recent QA pairs
memo.mid_term_memory.sessions          # Consolidated sessions
```

### 1.4 MemoryOS vs Other Baselines

| Feature | MemoryOS | Mem0 | LightMem | A-mem |
|---------|----------|------|----------|-------|
| Memory tiers | 3 (ST/MT/LT) | 1 (flat) | 1 (embedding) | 1 (notes) |
| Extraction | LLM-based | `infer=True` | LLMLingua-2 | K/G/X auto |
| Heat/importance | ✅ | ❌ | ❌ | Evolution |
| User profile | ✅ | ❌ | ❌ | ❌ |
| Storage | JSON files | Qdrant | Qdrant | ChromaDB |

---

## 2. Understanding the Hangman Agent Architecture

### 2.1 Base Agent Interface

All agents inherit from `BaseAgent` (`src/hangman/agents/base_agent.py`):

```python
class BaseAgent(ABC):
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.workflow: StateGraph = self._build_workflow()

    @abstractmethod
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """Run one turn, return {'response': str, 'thinking': str}"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get internal LangGraph state"""
        pass

    @abstractmethod
    def get_private_state(self) -> str:
        """Get private state as string for logging/evaluation"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset for new trial"""
        pass
```

### 2.2 Memory Agent Pattern (from Mem0Agent/LightMemAgent)

Memory-based agents follow this pattern:

1. **Constructor** (`__init__`):
   - Accept `llm_provider`, config path/dict, `session_id`, branching params
   - Initialize memory system with session-specific namespacing
   - Create sliding window (`deque`) for recent messages
   - Store config for cloning

2. **Agent Node** (`_agent_node`):
   - Update sliding window with incoming messages
   - Retrieve relevant memories for the query
   - Build system prompt with retrieved memories
   - Call LLM provider for response
   - Store interaction in memory system

3. **SCT Forking Support**:
   - `get_session_config()`: Return config for creating branch agents
   - `clone_memories_from(parent)`: Copy memories to branch
   - `get_sliding_window_state()`: Return window for seeding branches

### 2.3 Agent Registration

Agents are registered in `src/hangman/agents/__init__.py`:

```python
from hangman.agents.memoryos_agent import MemoryOSAgent  # Add this

__all__ = [
    ...,
    "MemoryOSAgent",  # Add this
]
```

And factory pattern for instantiation in `run_sct_hangman.py`.

---

## 3. SCT Engine Integration

### 3.1 How SCT Works (`engine_sct_hangman.py`)

The Self-Consistency Test:
1. Plays pre-fork turns with a deterministic player
2. At `t_fork`, captures agent state (memories, sliding window)
3. Creates **branch agents** for each candidate secret word
4. Asks each branch "Is [word] your secret?" (yes/no)
5. Evaluates consistency of answers

### 3.2 Branch Agent Requirements

For SCT forking to work, MemoryOSAgent must support:

```python
# In engine_sct_hangman.py, _run_branch():
if isinstance(self.agent, MemoryOSAgent):
    config = self.agent.get_session_config()
    branch_agent = MemoryOSAgent(
        branch_id=str(branch_id),
        parent_session_id=config["session_id"],
        **config
    )
    # Clone pre-fork memories to the branch
    branch_agent.clone_memories_from(self.agent.memo)
    
    # Seed the sliding window with pre-fork messages
    pre_fork_window = self.agent.get_sliding_window_state()
    for msg in pre_fork_window:
        branch_agent._window.append(msg)
```

---

## 4. Files to Create

### 4.1 `src/hangman/agents/memoryos_agent.py`

Main agent implementation following the Mem0Agent pattern:

```python
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
from hangman.providers.llmprovider import LLMProvider

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
                    knowledge = "\n".join(f"- {k}" for k in knowledge)
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
                    knowledge = "\n".join(f"- {k}" for k in knowledge)
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
```

### 4.2 `src/hangman/prompts/memoryos_agent.py`

```python
"""
Prompts for MemoryOSAgent.

MemoryOS provides hierarchical memories across three tiers:
- Short-term: Recent QA pairs
- Mid-term: Consolidated interaction segments
- Long-term: User profile and assistant knowledge
"""

MAIN_SYSTEM_PROMPT = """You are a helpful assistant with hierarchical long-term memory.

# INSTRUCTIONS
You have access to retrieved memories from three tiers:
1. **Recent Interactions**: Your most recent exchanges with the user
2. **User Profile**: Accumulated knowledge about the user's preferences and traits
3. **Learned Knowledge**: Facts and information you've learned across sessions

Use these memories to:
- Maintain consistency with your previous statements and decisions
- Recall relevant context from prior interactions
- Avoid contradicting yourself or repeating information unnecessarily
- Inform your responses naturally without explicitly listing memories

If no relevant memory applies, proceed as usual while staying consistent with any task instructions.

# RETRIEVED MEMORIES
{memories}
"""
```

### 4.3 `config/memoryos_config_gptoss_20b.yaml`

```yaml
# MemoryOS Configuration for GPT-OSS-20B via OpenRouter
#
# Note: API key is read from OPENROUTER_API_KEY environment variable

llm_model: openai/gpt-oss-20b

# Memory tier capacities
short_term_capacity: 10
mid_term_heat_threshold: 5
retrieval_queue_capacity: 7
long_term_knowledge_capacity: 100

# Data storage (overridden per-session in agent code)
data_storage_path: ./memoryos_data

# Optional embedding model (if MemoryOS supports custom embeddings)
# embedding_model_name: all-MiniLM-L6-v2
```

---

## 5. Files to Edit

### 5.1 `src/hangman/agents/__init__.py`

Add the MemoryOSAgent import and registration:

```python
# Add import
from hangman.agents.memoryos_agent import MemoryOSAgent

# Add to __all__
__all__ = [
    ...,
    "MemoryOSAgent",
]

# Add to factory (if used)
def create_agent(...):
    ...
    elif agent_name == "MemoryOSAgent":
        return MemoryOSAgent(llm_provider=llm_provider)
```

### 5.2 `src/hangman/engine_sct_hangman.py`

Add MemoryOSAgent handling in `_build_candidates()` and `_run_branch()`:

```python
from hangman.agents.memoryos_agent import MemoryOSAgent

# In _build_candidates() reveal secret fork:
elif isinstance(self.agent, MemoryOSAgent):
    config = self.agent.get_session_config()
    branch_agent = AgentClass(
        branch_id="reveal",
        parent_session_id=config["session_id"],
        **config
    )
    branch_agent.clone_memories_from(self.agent.memo)
    pre_fork_window = self.agent.get_sliding_window_state()
    for msg in pre_fork_window:
        branch_agent._window.append(msg)

# In _run_branch():
elif isinstance(self.agent, MemoryOSAgent):
    config = self.agent.get_session_config()
    branch_agent = AgentClass(
        branch_id=str(branch_id),
        parent_session_id=config["session_id"],
        **config
    )
    branch_agent.clone_memories_from(self.agent.memo)
    pre_fork_window = self.agent.get_sliding_window_state()
    for msg in pre_fork_window:
        branch_agent._window.append(msg)
```

### 5.3 `run_sct_hangman.py`

Add session_id injection for MemoryOSAgent:

```python
# In _run_trial_job() and sequential loop:
if class_name in ["Mem0Agent", "AMemAgent", "LettaAgent", "LightMemAgent", "MemoryOSAgent"]:
    raw_kwargs["session_id"] = trial_session_id

# Post-instantiation update:
elif isinstance(agent, MemoryOSAgent):
    agent._thread_id = f"memoryos_main__{agent.session_id}"
```

### 5.4 `config/hangman_sct_gptoss_run.yaml`

Add MemoryOSAgent to the agents list:

```yaml
agents:
  - MemoryOSAgent:
      name: memoryos_agent
      memoryos_config_path: ./config/memoryos_config_gptoss_20b.yaml
```

---

## 6. Dependencies

Add to `pyproject.toml` or `requirements.txt`:

```toml
# PyPI package
memoryos-pro = "^0.1.0"

# Or from GitHub (latest)
# memoryos @ git+https://github.com/BAI-LAB/MemoryOS.git
```

---

## 7. Testing Checklist

- [ ] **Unit test**: Interactive CLI test (`python -m hangman.agents.memoryos_agent`)
- [ ] **Integration test**: Single trial with MemoryOSAgent
- [ ] **SCT test**: Full 50-trial run with branching
- [ ] **Memory inspection**: Verify `get_private_state()` shows all tiers
- [ ] **Cloning test**: Verify branch agents receive pre-fork memories
- [ ] **Evaluation**: Compare SCT metrics with other memory baselines

---

## 8. Expected Challenges

1. **File-based storage isolation**: MemoryOS uses JSON files. Need unique `data_storage_path` per session/branch to avoid collisions in parallel runs.

2. **Memory cloning for SCT**: MemoryOS doesn't have a native "clone" API. Implementation copies short-term memories via `add_memory()`. May need to copy files directly for mid/long-term.

3. **Heat threshold timing**: With short trials (t_fork=4), mid-term/long-term may not activate. Short-term memory should still work.

4. **OpenRouter compatibility**: MemoryOS expects OpenAI API. Use `openai_base_url="https://openrouter.ai/api/v1"` and set model names with `openai/` prefix.

---

## 9. File Summary

| Action | File | Purpose |
|--------|------|---------|
| **Create** | `src/hangman/agents/memoryos_agent.py` | Main agent implementation |
| **Create** | `src/hangman/prompts/memoryos_agent.py` | System prompt with memory formatting |
| **Create** | `config/memoryos_config_gptoss_20b.yaml` | MemoryOS configuration |
| **Create** | `config/memoryos_config_qwen3_32b.yaml` | Alternative model config |
| **Edit** | `src/hangman/agents/__init__.py` | Register MemoryOSAgent |
| **Edit** | `src/hangman/engine_sct_hangman.py` | Add SCT branching support |
| **Edit** | `run_sct_hangman.py` | Add session_id injection |
| **Edit** | `config/hangman_sct_gptoss_run.yaml` | Add agent to run config |
| **Edit** | `pyproject.toml` | Add memoryos-pro dependency |

---

## 10. References

- [MemoryOS GitHub](https://github.com/BAI-LAB/MemoryOS)
- [MemoryOS Paper (arXiv)](https://arxiv.org/abs/2506.06326)
- [MemoryOS Documentation](https://bai-lab.github.io/MemoryOS/docs)
- Existing baselines: `mem0_agent.py`, `lightmem_agent.py`, `amem_agent.py`


