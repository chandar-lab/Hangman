# Working Memory Structure for the Agent

The working memory is divided into **three core sections**, inspired by cognitive psychology and agent design literature.  
This structure is minimal yet complete, supporting long-horizon coherence, efficient reasoning, and clear filing rules for the agent.

---

## 1. Goals / Plans

**What to store:**  
- Current overarching goal.  
- Subgoals or milestones.  
- Planned steps or strategies for achieving them.  

**Rationale:**  
In human cognition, **goal maintenance** in working memory guides attention and action selection. In LLM-based agents, explicitly tracking **subgoals as memory chunks** helps maintain focus, decide when to replace obsolete items, and summarize completed tasks.  
> “HiAgent prompts LLMs to formulate subgoals… enables LLMs to decide proactively to replace previous subgoals with summarized observations…”  
— Hu et al., *HiAgent: Hierarchical Memory Management for LLM-based Agents* (2024) [arXiv:2408.09559](https://arxiv.org/abs/2408.09559)

---

## 2. Facts / Knowledge (Semantic and Episodic Memory)

**What to store:**  
- **Semantic facts**: stable knowledge about the user, environment, domain, constraints.  
- **Episodic facts**: specific events or experiences relevant to the current or future tasks.  
- **Summaries**: condensed versions of larger knowledge chunks to save space.  

**Rationale:**  
This section functions like the **episodic buffer** in Baddeley’s model, integrating and holding information from different sources temporarily while linking to long-term memory. Chunking theory suggests compressing related details into meaningful units improves retention and reduces capacity load.  
> “The episodic buffer is a limited-capacity store that binds together information from a variety of sources…”  
— Baddeley, *The episodic buffer: a new component of working memory?* (2000)  
> “Working memory has a capacity of about four chunks… chunking… enables grouping information into meaningful units.”  
— Cowan, *The Magical Mystery Four* (2001)

---

## 3. Active Knowledge (Reasoning Outputs & Ephemeral Notes)

**What to store:**  
- Current observations actively used in decision-making.  
- Intermediate reasoning outputs from the agent’s thought process.  
- Ephemeral notes or hypotheses, which can be discarded once resolved.  

**Rationale:**  
In cognitive architectures, the **active memory store** holds the most relevant, time-sensitive information for the next reasoning or action step. Including ephemeral reasoning states here mirrors how the **central executive** coordinates immediate problem solving in human working memory.  
> “Working memory is the system responsible for the transient holding and processing of new and already stored information…”  
— Baddeley, *Working Memory: Theories, Models, and Controversies* (2012)  
> “Active memory is the subset of working memory currently in use for immediate tasks, constantly updated as the situation changes.”  
— Laird et al., *Soar Cognitive Architecture* (2017)

---