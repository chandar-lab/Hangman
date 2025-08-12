
## Baseline Agents: Experimental Grid

This document organizes the baseline agents along two axes:

**1. Agent Autonomy**
	- **Full Autonomy:** Agent decides when and how to manage memory (e.g., ReAct).
	- **Forced Workflow:** Agent is required to perform a memory operation at every turn (e.g., ReaDis* agents).

**2. Memory Management Strategy**
	- **Update:** Memory is updated incrementally.
	- **Patch:** Memory is patched (modified in-place, possibly partially).
	- **Overwrite:** Memory is fully overwritten each time.
	- **None/Self-prompt:** No memory; agent must self-prompt or act statelessly.

### Baseline Grid

|                    | Update               | Patch                 | Overwrite              | None/Self-prompt      |
|--------------------|----------------------|-----------------------|------------------------|-----------------------|
| **Full Autonomy**  | ReAct-Update         | ReAct-Patch           | ReAct-Overwrite        | ReAct-NoMemory        |
| **Forced Workflow**| ReaDisUpdAct         | ReaDisPatAct          | ReaDisOveAct           | Stateless-Forced      |

#### Descriptions

- **ReAct-Update:** ReAct agent with full autonomy, can update memory at will.
- **ReAct-Patch:** ReAct agent with full autonomy, can patch memory at will.
- **ReAct-Overwrite:** ReAct agent with full autonomy, can overwrite memory at will.
- **ReAct-NoMemory:** ReAct agent with no memory, relies on self-prompting.
- **ReaDisUpdAct:** Forced to update memory every turn.
- **ReaDisPatAct:** Forced to patch memory every turn.
- **ReaDisOveAct:** Forced to overwrite memory every turn.
- **Stateless-Forced:** Forced to act without memory (stateless, forced workflow).

### Additional Baselines

- **Vanilla:** Stateless agent, no memory, no reasoning chain.
- **Public CoT:** Stateless agent, chain-of-thought reasoning, no memory.
- **ReAct + Heuristic Summarization:** Hybrid; ReAct agent with autonomous memory management, but memory is summarized heuristically.

### Experimental Recommendations

To fully explore the autonomy × memory management space, implement at least one agent per cell in the above grid, plus the stateless and hybrid baselines. This enables a clear analysis of how autonomy and memory management interact to affect performance.
