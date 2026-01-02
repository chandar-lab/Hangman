### **4.1. Main Result: The Impossibility of Stateless Consistency**
**Goal:** The "Money Shot." Show that standard agents and vector-memory agents fail systematically, while your solution (Private Scratchpad) succeeds.

* **The Metric:** SCT Accuracy (Percentage of trials falling into **Case 1** only).
* **The Visualization:** A large Table or Grouped Bar Chart.
    * **Rows:** Baselines (Vanilla, Amem, Mem0, LightMem, Summary, Private CoT, **Ours**).
    * **Columns:** Model Families (Qwen 32B/320B, GPT-OSS 20B/120B) $\times$ Domains (Hangman, Diagnosis).
* **The Takeaway:**
    * Demonstrate that Vanilla and RAG-based methods (Mem0/LightMem) have near-random performance (or significantly below ceiling).
    * Show that **Ours** approaches the "Upper Bound" of Private CoT (Full History) but with constant context size.

### **4.2. Failure Mode Analysis: Evidence of Superposition**
**Goal:** Validate the *Impossibility Theorem*. You hypothesized that stateless agents are in a "superposition" of possible secrets. If this is true, they should overwhelmingly fail by **confirming multiple secrets** (Case 2), not by remaining silent (Case 3).

* **The Metric:** Breakdown of Outcomes (Success vs. Superposition vs. Silence).
* **The Visualization:** Stacked Bar Chart (normalized to 100%).
    * **Green:** Case 1 (Unique/Correct).
    * **Red:** Case 2 (Self-Inconsistent / Confirmed Multiple).
    * **Gray:** Case 3 (Rejection / Confirmed None).
* **The Narrative:**
    * *"Stateless agents display high 'compliance': they tend to agree with the user's fork. This empirically validates the superposition argument—without a fixed anchor, the conditional probability $P(Yes | History + Candidate)$ is high for all valid candidates."*
    * *Contrast:* Show that RAG baselines (Mem0) often fall into Case 2 because they retrieve the *concept* of the game but not the *specific exclusions*.

### **4.3. The "Scale is Not a Fix" Ablation**
**Goal:** Prove that simply making the model bigger does not create a hidden state.

* **The Metric:** Delta in Accuracy between Small (20B/32B) and Large (120B/320B) models.
* **The Visualization:** Side-by-side comparison or a scatter plot.
    * X-axis: Model Size.
    * Y-axis: SCT Accuracy.
* **The Narrative:**
    * *"While larger models have better reasoning (Diagnosis score improves slightly), they suffer the same structural amnesia. Scaling laws do not emerge for State Retention in POCAs."*

### **4.4. Efficiency Analysis: The Cost of Consistency**
**Goal:** Justify why we can't just use "Private CoT with Full History" (the brute force method).

* **The Metric:** Token consumption per turn vs. Dialogue Length.
* **The Visualization:** Line Chart.
    * X-axis: Turn number ($t$).
    * Y-axis: Total tokens processed.
    * **Lines:**
        * Line A (Steep linear/quadratic slope): Private CoT (Full History).
        * Line B (Flat/Constant slope): **Ours** (Scratchpad).
* **The Narrative:**
    * *"Our method achieves comparable consistency to the Full History baseline but with $O(1)$ memory complexity relative to dialogue length (assuming constant scratchpad size), whereas Full History is $O(N)$."*

### **4.5. Qualitative Case Study: Why Vector Memory Fails**
#### **Case Study 1: The "Anchor" Mechanism (Why Ours Succeeds vs. Baselines)**
**Research Question:** Does the scratchpad actually serve as a "hard anchor" that prevents the agent from drifting into superposition?
**What to display:**
A side-by-side comparison of the **Private State** at $t_{fork}$ for your agent vs. a Baseline (Mem0 or Vanilla), given the *exact same* public history.

#### **Case Study 2: The "Semantic Retrieval" Gap (Why RAG/Mem0 Fails)**
**Research Question:** Why do Memory-based agents (Mem0/Amem) fail even though they "have memory"?
**What to display:**
A Diagnosis Simulator example where the failure is due to **ignoring a negative constraint**.

#### **Case Study 3: The "Ambiguous Update" (Why Ours Fails)**
**Research Question:** When our Scratchpad agent fails, is it a failure of *architecture* or *execution*?

#### **Case Study 4: The "Lucky Guess" (Why Baselines Succeed)**
**Research Question:** When Vanilla/Mem0 succeeds (Case 1), did it actually "know" the secret?

### **4.6. Distrubution of Secrets
---

### **Summary of the Story You Are Telling**

1.  **Table 1:** Everyone fails except State-aware models.
2.  **Figure 1 (Stacked Bar):** They fail because they agree with everything (Superposition).
3.  **Figure 2 (Scale):** Buying a bigger GPU doesn't fix it.
4.  **Figure 3 (Efficiency):** Re-reading the whole chat is too expensive; our scratchpad is efficient.
5.  **Box 1 (Qualitative):** Vectors are for search, not for holding a variable.

Does this structure resonate with the data you are currently seeing?

