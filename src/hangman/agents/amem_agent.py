# pip install git+https://github.com/agiresearch/A-mem.git
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

# A-mem system
from agentic_memory.memory_system import AgenticMemorySystem  # from the repo

from hangman.prompts.amem_agent import MAIN_SYSTEM_PROMPT


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    last_question: str
    last_system_prompt: str
    retrieved_notes: List[Dict[str, Any]]

class AMemAgent(BaseAgent):
    """
    Paper-faithful A-mem baseline:
      - On each turn: add a new note from the latest HumanMessage (A-mem auto-generates K/G/X, embeds, links, evolves).
      - Retrieval: search top-k notes for the latest user query.
      - Generation: System(MEMORY NOTES) + last m Human/AI messages.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        amem_model_name: str = "all-MiniLM-L6-v2",  # Embedding model
        amem_llm_backend: str = "openrouter",           # or "ollama"
        amem_llm_model: str = "openai/gpt-oss-20b",
        session_id: str = "amem_session_1",
        m_recent: int = 30,
        k_retrieve: int = 8,
    ):
        self.session_id = session_id
        self.m_recent = int(m_recent)
        self.k_retrieve = int(k_retrieve)
        
        # Store A-mem configuration for branching/cloning
        self.amem_model_name = amem_model_name
        self.amem_llm_backend = amem_llm_backend
        self.amem_llm_model = amem_llm_model

        # Initialize A-mem memory system
        # If using OpenRouter via OpenAI SDK:
        #   export OPENAI_BASE_URL=https://openrouter.ai/api/v1
        #   export OPENAI_API_KEY=$OPENROUTER_API_KEY
        self.amem = AgenticMemorySystem(
            model_name=amem_model_name,
            llm_backend=amem_llm_backend,
            llm_model=amem_llm_model,
        )

        # Sliding window for composing messages to the LLM (not for A-mem ingestion)
        self._window: deque = deque([], maxlen=self.m_recent)

        super().__init__(llm_provider=llm_provider)
        self.reset()

    def _build_workflow(self) -> StateGraph:
        wf = StateGraph(AgentState)
        wf.add_node("agent", self._agent_node)
        wf.set_entry_point("agent")
        return wf.compile(checkpointer=MemorySaver())

    # ---------- Core node ----------
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        incoming: List[BaseMessage] = state.get("messages", [])

        # Update conversation window (for model history composition)
        for msg in incoming:
            if isinstance(msg, (HumanMessage, AIMessage)):
                self._window.append(msg)

        # Latest user text is the "interaction content" to turn into a note
        latest_user = ""
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                latest_user = str(msg.content)
                break

        # 1) NOTE CONSTRUCTION (+ auto K/G/X/embedding), LINK GEN, EVOLUTION
        if latest_user:
            # A-mem auto-adds: generates keywords/tags/context, embeds, links, evolves
            try:
                _note_id = self.amem.add_note(content=latest_user)
            except Exception:
                pass  # don't crash the turn if A-mem fails

        # 2) RETRIEVAL for generation
        retrieved = []
        try:
            results = self.amem.search(latest_user or "", k=self.k_retrieve) or []
            # normalize shape into a list of dicts we can print
            for r in results:
                # Common fields from README: id, content, keywords, tags, (maybe score/context)
                retrieved.append({
                    "id": r.get("id"),
                    "content": r.get("content") or "",
                    "keywords": r.get("keywords") or [],
                    "tags": r.get("tags") or [],
                    "category": r.get("category") or "",
                    "context": r.get("context") or "",
                    "score": r.get("score"),
                })
        except Exception:
            retrieved = []

        # Build a compact notes block for the system prompt
        if retrieved:
            lines = []
            for n in retrieved:
                kw = ", ".join(n["keywords"]) if n["keywords"] else "-"
                tg = ", ".join(n["tags"]) if n["tags"] else "-"
                cx = n["context"][:300] + ("..." if n["context"] and len(n["context"]) > 300 else "")
                ct = n["content"][:400] + ("..." if n["content"] and len(n["content"]) > 400 else "")
                lines.append(f"- Content: {ct}\n  Keywords: {kw}\n  Tags: {tg}\n  Context: {cx}")
            notes_block = "\n".join(lines)
        else:
            notes_block = "(none)"

        system_prompt = MAIN_SYSTEM_PROMPT.format(notes_block=notes_block)

        # 3) Compose messages like your README: System(memory) + last m turns
        messages_for_model: List[BaseMessage] = [SystemMessage(content=system_prompt)] + list(self._window)

        # 4) Call your LLM provider
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
            "last_question": latest_user,
            "last_system_prompt": system_prompt,
            "retrieved_notes": retrieved,
        }

    # ---------- BaseAgent interface ----------
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        cfg = {"configurable": {"thread_id": f"amem__{self.session_id}"}}
        final_state = self.workflow.invoke({"messages": messages}, config=cfg)
        final_response = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                final_response = m.content
                break
        return {"response": final_response, "thinking": final_state.get("thinking", "")}

    def get_state(self) -> Dict[str, Any]:
        snap = self.workflow.get_state({"configurable": {"thread_id": f"amem__{self.session_id}"}})
        return snap.values if snap else {}

    def get_private_state(self) -> str:
        # summarize retrieved last turn for logging
        st = self.get_state()
        retrieved = st.get("retrieved_notes", []) or []

        if retrieved:
            lines = []
            for n in retrieved:
                kw = ", ".join(n["keywords"]) if n["keywords"] else "-"
                tg = ", ".join(n["tags"]) if n["tags"] else "-"
                cx = n["context"][:300] + ("..." if n["context"] and len(n["context"]) > 300 else "")
                ct = n["content"][:400] + ("..." if n["content"] and len(n["content"]) > 400 else "")
                lines.append(f"- Content: {ct}\n  Keywords: {kw}\n  Tags: {tg}\n  Context: {cx}")
            notes_block = "\n".join(lines)
        else:
            notes_block = "(none)"

        return notes_block

    def reset(self) -> None:
        self._window.clear()
        empty: AgentState = AgentState(messages=[], thinking="")
        self.workflow.update_state({"configurable": {"thread_id": f"amem__{self.session_id}"}}, empty)

    # ---------- SCT Support Methods ----------
    def get_session_config(self) -> Dict[str, Any]:
        """
        Return configuration needed to create a branch agent.
        Used by SCT engine to instantiate branch agents with the same settings.
        
        Returns:
            Dict with llm_provider, amem config, session params, and parent amem instance.
        """
        return {
            "llm_provider": self.llm_provider,
            "amem_model_name": self.amem_model_name,
            "amem_llm_backend": self.amem_llm_backend,
            "amem_llm_model": self.amem_llm_model,
            "session_id": self.session_id,
            "m_recent": self.m_recent,
            "k_retrieve": self.k_retrieve,
            "parent_amem": self.amem,  # Pass parent's amem instance for cloning
        }
    
    def clone_memories_from(self, parent_amem: Any) -> None:
        """
        Clone memories from parent's AgenticMemorySystem to this agent's memory.
        
        This retrieves all notes from the parent's A-mem instance and adds them
        to this agent's A-mem, preserving the pre-fork memory state for branch agents.
        
        Args:
            parent_amem: The parent agent's AgenticMemorySystem instance to clone from
        """
        if parent_amem is None:
            return
        
        try:
            # A-mem stores notes in a 'memories' dict: {uuid: MemoryNote, ...}
            if hasattr(parent_amem, 'memories') and isinstance(parent_amem.memories, dict):
                parent_memories = parent_amem.memories
                
                # Add each parent note to this agent's memory
                for note_id, note in parent_memories.items():
                    try:
                        # Add note content to this agent's A-mem
                        # This will create a new note with new embeddings and links
                        if hasattr(note, 'content') and note.content:
                            self.amem.add_note(content=note.content)
                    except Exception:
                        # Skip notes that fail to add (continue with others)
                        continue
        except Exception:
            # If cloning fails, branch continues with empty memory
            # and relies on sliding window for context
            pass
    
    def get_sliding_window_state(self) -> List[BaseMessage]:
        """
        Return the current sliding window messages.
        Used by SCT engine to seed branch agents with pre-fork conversation state.
        
        Returns:
            List of BaseMessage objects in the sliding window
        """
        return list(self._window)

    
# src/hangman/agents/amem_main.py




def main():
    # Optional: make OpenRouter work with libs that expect OpenAI env
    # os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    # os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))

    # 1) Load config (only needed if your load_llm_provider reads from it)
    CONFIG_PATH = "config/config.yaml"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        _ = yaml.safe_load(f)

    # 2) Initialize your OpenRouter-backed LLM provider
    #    Match the provider name you use elsewhere (e.g., "qwen3_235b_openrouter" or "gpt_oss_20b_openrouter")
    try:
        llm = load_llm_provider(CONFIG_PATH, provider_name="gpt_oss_20b_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        return

    # 3) Initialize A-mem agent
    agent = AMemAgent(
        llm_provider=llm,
        amem_model_name="all-MiniLM-L6-v2",   # embedding model for Chroma/AgenticMemorySystem
        amem_llm_backend="openrouter",         
        amem_llm_model="openai/gpt-oss-20b",         # for A-mem’s internal analysis/evolution
        session_id="amem_session_1",
        m_recent=10,
        k_retrieve=8,
    )
    print("🤖 AMemAgent is ready. Type 'quit', 'exit', or 'q' to end.")

    # 4) Interactive loop
    messages = []
    while True:
        try:
            user_input = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Ending session.")
            break

        messages.append(HumanMessage(content=user_input))
        output: ModelOutput = agent.invoke(messages)
        messages.append(AIMessage(content=output["response"]))

        print("\n---ANSWER---")
        print(f"AI: {output['response']}")
        if output.get("thinking"):
            print("\n---THINKING TRACE---")
            print(output["thinking"])

        print("\n---A-MEM RETRIEVED NOTES (last turn)---")
        print(agent.get_private_state())

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()