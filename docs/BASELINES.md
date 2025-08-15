# Baselines

This document describes the baselines used in our evaluation of **Private State Interactive Tasks (PSIT)** such as Hangman and 20 Questions.  
Baselines are chosen to cover the spectrum of **stateless models**, **reasoning traces**, and **private state management** implemented in both **agent** and **workflow** frameworks.  
They include both realistic existing approaches (engineering and literature) and hypothetical constructions illustrating theoretical limits.

---

## 1. Stateless LLM
**Description:**  
A reasoning LLM whose intermediate reasoning tokens (scratchpad) are **discarded** once the final response is produced.  
On each turn, the model sees:
- System prompt (no private state).
- Public conversation history (user and assistant’s final messages only).

**Purpose:**  
Serves as the pure “no state” baseline, demonstrating failure modes when no persistent private information is available.

---

## 2. Public CoT LLM
**Description:**  
The LLM produces a chain-of-thought (CoT) reasoning trace **visible** to the user.  
This reasoning is appended to the public conversation history and is accessible to both model and user in subsequent turns.

**Purpose:**  
Tests whether exposing reasoning to the user (but still lacking hidden state) can improve consistency or task performance.

---

## 3. Private CoT LLM
**Description:**  
The LLM produces a CoT reasoning trace **hidden from the user** but preserved internally between turns.  
This “private” reasoning is appended to the conversation history **agent-side only**.

**Purpose:**  
Simulates a limited form of private state via retained reasoning traces, without introducing explicit structured memory.

---

## 4. ReAct (no tools)  
**Description:**  
A ReAct-style agent framework with no available tools.  
On each turn, the model:
- Receives the system prompt (no working memory).
- Sees the public conversation history (user + assistant final messages only).
- Generates a reasoning trace for the current turn, which is discarded after producing the final message.

**Equivalence to Stateless LLM:**  
Without tools or private memory, this setup is behaviorally identical to the **Stateless LLM** baseline.  
We include it in the taxonomy for completeness but treat it as equivalent in results.

---

## 5. ReAct + Update Memory Tool
**Description:**  
A ReAct agent with a single tool: `update_memory`.  
- The working memory is included in the system message each turn.
- After reasoning, the agent may call `update_memory` to modify the private working memory before producing its final message.

**Purpose:**  
Tests whether explicit memory updates via tool calls can overcome PSIT limitations.

---

## 6. Update Memory Workflow
**Description:**  
A structured workflow (no agent autonomy) with two fixed steps per turn:
1. **Response LLM** generates an answer given the public history + current private working memory.
2. **Memory Update LLM** updates the working memory given the conversation and previous memory.

**Purpose:**  
Contrasts agentic memory updates with a fixed deterministic flow.

---

## 7–9. Memory Update Strategies
For both the **ReAct + Update Memory Tool** and the **Update Memory Workflow** baselines, we test three strategies for modifying private working memory:

### 7. Overwrite Memory 
- The entire memory is replaced each update.

### 8. Delete/Insert Memory Items 
- The model removes outdated entries and inserts new ones.

### 9. Patch Memory
- The model outputs a textual patch (diff) applied to the memory, preserving unchanged parts.


## Summary Table

| ID  | Name                                      | Framework | Private State | Reasoning Visibility | Memory Update Method | Status               |
|-----|-------------------------------------------|-----------|---------------|----------------------|----------------------|----------------------|
| 1   | Stateless LLM                             | None      | ❌            | Hidden (discarded)   | N/A                  | Done                 |
| 2   | Public CoT LLM                            | None      | ❌            | Public               | N/A                  | Done                 |
| 3   | Private CoT LLM                           | None      | ✅            | Private              | N/A                  | Done                 |
| 4   | ReAct (no tools)                          | Agent     | ❌            | Hidden (discarded)   | N/A (≡ Stateless LLM)| Done                 |
| 5   | ReAct + Update Memory Tool (Overwrite)    | Agent     | ✅            | Private              | Overwrite            | Done    |
| 6   | ReAct + Update Memory Tool (Delete/Insert)| Agent     | ✅            | Private              | Delete/Insert        | Done                 |
| 7   | ReAct + Update Memory Tool (Patch)        | Agent     | ✅            | Private              | Patch                | Done    |
| 8   | Update Memory Workflow (Overwrite)        | Workflow  | ✅            | Private              | Overwrite            | To Be Refactored     |
| 9   | Update Memory Workflow (Delete/Insert)    | Workflow  | ✅            | Private              | Delete/Insert        | To Be Refactored     |
| 10  | Update Memory Workflow (Patch)            | Workflow  | ✅            | Private              | Patch                | To Be Refactored     |
