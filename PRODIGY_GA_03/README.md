# PRODIGY_GA_03: Text Generation using Character-Level Markov Chains

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Prodigy InfoTech](https://img.shields.io/badge/Prodigy_InfoTech-Generative_AI-blueviolet?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-03-success?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/Dependencies-Standard_Library-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Objective](#2-objective)
3. [What is a Markov Chain?](#3-what-is-a-markov-chain)
4. [How Markov Chains Generate Text](#4-how-markov-chains-generate-text)
5. [Architecture & Workflow](#5-architecture--workflow)
6. [Algorithm & Mathematical Formulation](#6-algorithm--mathematical-formulation)
7. [Technologies & Tools](#7-technologies--tools)
8. [Project Structure](#8-project-structure)
9. [Installation & Setup](#9-installation--setup)
10. [How to Run](#10-how-to-run)
11. [Example Input](#11-example-input)
12. [Example Output & Order Comparison](#12-example-output--order-comparison)
13. [Testing & Quality Assurance](#13-testing--quality-assurance)
14. [Limitations](#14-limitations)
15. [Future Improvements](#15-future-improvements)
16. [Internship Information](#16-internship-information)
17. [Viva / Interview Preparation Q&A](#17-viva--interview-preparation-qa)

---

## 1. Project Overview

**PRODIGY_GA_03** is an end-to-end character-level text generation engine built using stochastic **Markov Chains**. Developed as part of the **Prodigy InfoTech Generative AI Internship (Task-03)**, this project demonstrates foundational probabilistic sequence modeling, n-gram state transitions, weighted random sampling, and temperature-controlled text synthesis.

The project is implemented entirely using Python's standard library for zero-dependency portability and academic clarity, making it easy to explain in technical vivas and code reviews.

---

## 2. Objective

- Build a **character-level Markov Chain** model from scratch in Python.
- Learn empirical character-transition probability distributions from a local corpus (`data/training_text.txt`).
- Synthesize new, coherent text sequences mimicking the vocabulary and syntactic structures of the source material.
- Provide configurable parameters:
  - **Markov Order ($k$)**: Length of the context window state ($k = 1, 2, 3, 4, 5$).
  - **Starting Seed**: Customizable character prefix or prompt.
  - **Generation Length**: Number of characters to generate.
  - **Temperature ($T$)**: Control generation randomness / creativity.
  - **Multi-sample Generation**: Generate multiple candidate variations in one run.
- Handle edge cases safely (empty datasets, unseen seeds, sink/dead-end states).
- Provide automated unit and integration tests.

---

## 3. What is a Markov Chain?

A **Markov Chain** is a stochastic process that transitions between states in a state space according to probabilistic rules. The fundamental defining principle is the **Markov Property (Memorylessness)**:

$$\mathbb{P}(X_{t+1} = s \mid X_t = x_t, X_{t-1} = x_{t-1}, \dots, X_0 = x_0) = \mathbb{P}(X_{t+1} = s \mid X_t = x_t)$$

In simple terms: **the probability of moving to the next state depends solely on the current state, not on the entire history of preceding states.**

### Transition Matrix Example (Order 1)

If the vocabulary consists of characters $\{a, b, c\}$, the transition matrix $P$ contains probabilities $P_{ij} = \mathbb{P}(\text{next} = j \mid \text{current} = i)$:

| Current State | Next: 'a' | Next: 'b' | Next: 'c' |
|:---:|:---:|:---:|:---:|
| **'a'** | 0.10 | 0.60 | 0.30 |
| **'b'** | 0.40 | 0.00 | 0.60 |
| **'c'** | 0.50 | 0.50 | 0.00 |

---

## 4. How Markov Chains Generate Text

In **character-level** text generation with Markov order $k$:

1. **State Definition**: A state is an $n$-gram of $k$ consecutive characters (e.g., for $k=3$, `"art"`, `"gen"`, `"int"`).
2. **Training / Learning**: The model slides a window of length $k$ across the training corpus and counts how often each character follows each state:
   $$\text{Count}(\text{state} \to \text{char})$$
3. **Probability Calculation**:
   $$\mathbb{P}(\text{char} \mid \text{state}) = \frac{\text{Count}(\text{state} \to \text{char})}{\sum_{c'} \text{Count}(\text{state} \to c')}$$
4. **Generation / Sampling**:
   - Start with an initial state $S_0$ (either from a user seed or a sampled sentence starter).
   - Sample the next character $c_{next} \sim \mathbb{P}(\cdot \mid S_t)$ using weighted random selection.
   - Slide the window: $S_{t+1} = (S_t + c_{next})[-k:]$.
   - Repeat until the desired text length is reached.

---

## 5. Architecture & Workflow

```
+-------------------------------------------------------------------+
|                       Raw Training Corpus                         |
|                    (data/training_text.txt)                       |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                      Data Loader & Cleaner                        |
|       - Validates file existence and minimum length (len > k)     |
|       - Extracts vocabulary and sentence starting states          |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Markov Transition Engine                      |
|       - Slides window of size k across text                       |
|       - Builds nested map: transitions[state][next_char] = count  |
|       - Computes branching factors and state statistics           |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    Text Generation & Sampling                     |
|       - Resolves seed prompt (exact, prefix, or fallback)         |
|       - Applies temperature scaling: weight = count^(1/T)         |
|       - Probabilistically samples next character                  |
|       - Recovers gracefully from sink / dead-end states           |
+---------------------------------+---------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+--------------------+                           +--------------------+
|  CLI & Interactive |                           | Sample Output File |
|      Console       |                           |  (sample_output)   |
+--------------------+                           +--------------------+
```

---

## 6. Algorithm & Mathematical Formulation

### Algorithm: Character-Level Markov Text Generation

```
Algorithm: Markov_Train(Text, Order k)
Input  : Text string T of length N, Order k (k >= 1)
Output : Transition dictionary M, Start states list S

1. If N <= k, throw InsufficientDataError
2. Initialize M as empty map of maps
3. Initialize S as empty list
4. For i from 0 to N - k - 1:
     state = T[i : i + k]
     next_char = T[i + k]
     M[state][next_char] = M[state][next_char] + 1
     If i == 0 or T[i-2:i] in {". ", "! ", "? ", "\n"}:
         If state[0] is alphanumeric:
             Append state to S
5. Return M, S
```

```
Algorithm: Markov_Generate(M, S, Length L, Seed, Temperature T)
Input  : Transition map M, Start states S, Target Length L, Seed string, Temperature T
Output : Generated string G

1. Resolve initial state S_curr from Seed or sample from S
2. Set G = list(Seed or S_curr)
3. While length(G) < L:
     If S_curr not in M or M[S_curr] is empty:
         S_curr = sample_random(S)   // Dead-end recovery
         Append ' ' to G
     candidates = keys(M[S_curr])
     counts = values(M[S_curr])
     weights = [c ^ (1.0 / T) for c in counts]
     next_char = sample_weighted(candidates, weights)
     Append next_char to G
     S_curr = (S_curr + next_char)[-k:]
4. Return slice G[0 : L] as string
```

---

## 7. Technologies & Tools

- **Programming Language**: Python 3.8+
- **Modules Used (Zero external dependencies)**:
  - `collections.defaultdict`: Efficient nested transition graph storage.
  - `random`: Pseudo-random selection (`random.choices`, `random.choice`, `random.seed`).
  - `math`: Temperature scaling exponential transformations.
  - `argparse`: Comprehensive command-line interface.
  - `unittest`: Automated unit and integration testing.
  - `os`, `sys`, `pathlib`, `tempfile`: File system management and test isolation.

---

## 8. Project Structure

```
PRODIGY_GA_03/
│
├── markov_text_generator.py   # Core Markov Chain engine & CLI interface
├── test_markov_generator.py   # Comprehensive unit & integration test suite
├── generate_samples.py        # Verification script to produce sample_output.txt
├── requirements.txt           # Environment specifications
├── README.md                  # Complete technical documentation & Viva prep
│
├── data/
│   └── training_text.txt      # Rich, original training corpus on AI & ML (~8.8k chars)
│
└── output/
    └── sample_output.txt      # Verified generated text outputs & experiments
```

---

## 9. Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### Steps
1. Clone or navigate to the repository:
   ```bash
   cd PRODIGY_GA_03
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Verify requirements (uses standard library, zero external packages needed):
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. How to Run

### 1. Basic Generation
Run generation with default parameters (Order = 3, Length = 350):
```bash
python markov_text_generator.py
```

### 2. Custom Order, Seed, Length, and Temperature
```bash
python markov_text_generator.py --data data/training_text.txt --order 4 --length 400 --seed "Artificial intelligence" --temperature 0.8
```

### 3. Display Model Transition Statistics
```bash
python markov_text_generator.py --stats
```

### 4. Multi-Order Comparison (Orders 1 through 5)
Compare the impact of Markov order on coherence:
```bash
python markov_text_generator.py --compare --length 350 --seed "Artificial"
```

### 5. Multi-Sample Generation and Saving Output
Generate 3 samples and save to a file:
```bash
python markov_text_generator.py --order 4 --samples 3 --length 300 --output output/sample_output.txt
```

### 6. Interactive Mode
Launch an interactive shell to experiment with seeds and temperatures in real-time:
```bash
python markov_text_generator.py --interactive
```

---

## 11. Example Input

Training corpus excerpt from [`data/training_text.txt`](data/training_text.txt):
> *"Artificial intelligence is the simulation of human intelligence processes by machines... Generative artificial intelligence describes algorithms that can be used to create new content, including audio, synthetic code, images, text, simulations, and interactive media... In character-level generation, the model learns transition probabilities from a state of k characters to every possible successor character..."*

---

## 12. Example Output & Order Comparison

Real outputs generated from `data/training_text.txt`:

| Order ($k$) | Unique States | Avg Branching | Sample Generated Text | Coherence Level |
|:---:|:---:|:---:|:---|:---:|
| **$k=1$** | 51 | 172.29 | `Artificialull ar, by el rata itocequmadit csista. al on tere s ng t teg cadinte supranse...` | Pure character randomness |
| **$k=2$** | 474 | 18.54 | `Artificiallincipatifiencludenciplectionsupe arkov macy ing derspithatit oneuraccument...` | Syllables & morphemes |
| **$k=3$** | 1,908 | 4.60 | `Artificial inter-level Markov moderstatistimative represearning computative memorphe...` | English words forming |
| **$k=4$** | 3,681 | 2.39 | `Artificial interdiscrete sequence model memory. A first-order to another to drastic models...` | Grammatical phrases |
| **$k=5$** | 4,954 | 1.77 | `Artificial interdisciplinary scientifies the characteristic of a tokens regardless of...` | Highly fluent domain text |

Full verified experimental runs are saved in [`output/sample_output.txt`](output/sample_output.txt).

---

## 13. Testing & Quality Assurance

The test suite [`test_markov_generator.py`](test_markov_generator.py) verifies 10 automated test cases:

```bash
python -m unittest test_markov_generator.py -v
```

### Test Coverage Summary

| Test Case | Scenario | Description | Status |
|:---:|:---|:---|:---:|
| **TEST 1** | Normal Valid Input | Generates exact requested length of text | `PASSED` |
| **TEST 2** | Starting Seeds | Correctly initializes with custom character seeds | `PASSED` |
| **TEST 3** | Various Lengths | Validates outputs across lengths (10, 50, 200, 500, 1000) | `PASSED` |
| **TEST 4** | Unseen/Invalid Seed | Gracefully falls back without crashing on unseen characters | `PASSED` |
| **TEST 5** | Small/Zero Lengths | Handles boundary lengths (0, 1, 2) and raises on negative | `PASSED` |
| **TEST 6** | Insufficient Data | Raises `InsufficientDataError` for empty or too-short data | `PASSED` |
| **TEST 7** | File Training | Reads local text files and raises `FileNotFoundError` if missing | `PASSED` |
| **TEST 8** | Reproducibility | Deterministic outputs when `random_seed` is provided | `PASSED` |
| **TEST 9** | Temperature Scaling | Validates conservative ($T<1$) and creative ($T>1$) sampling | `PASSED` |
| **TEST 10** | Model Statistics | Computes unique states, vocabulary, and branching factors | `PASSED` |

---

## 14. Limitations

1. **Short Context Window**: Unlike Transformer models (e.g., GPT), a Markov chain only remembers the last $k$ characters and lacks long-range semantic memory.
2. **State Space Explosion**: As order $k$ increases, the number of potential states grows exponentially ($|\Sigma|^k$), requiring larger training data.
3. **Verbatim Reproduction at High Orders**: Very high orders ($k \ge 6$ on smaller datasets) lead to state transitions with branching factor 1, reproducing the training text verbatim.
4. **Out-of-Vocabulary Combinations**: Cannot synthesize characters or transition combinations never seen in the training data.

---

## 15. Future Improvements

- **Word-Level Markov Chain**: Implement word-level tokenization alongside character-level generation.
- **Backoff Smoothing (Katz Backoff / Kneser-Ney)**: Automatically fall back from order $k$ to $k-1$ when an unseen state is encountered.
- **Interactive Web Interface**: Build a lightweight web UI using Streamlit or Flask for real-time interactive generation.
- **Beam Search & Top-K Decoding**: Implement advanced decoding strategies.

---

## 16. Internship Information

- **Organization**: Prodigy InfoTech
- **Track**: Generative AI
- **Task**: Task-03 — Text Generation with Markov Chains
- **Repository**: `PRODIGY_GA_03`
- **Author**: Generative AI Intern

---

## 17. Viva / Interview Preparation Q&A

### Q1: What is the Markov Property?
> **Answer**: The Markov Property states that the future state of a system depends only upon its current state and not upon the historical sequence of prior events ($P(X_{t+1} \mid X_t, \dots, X_0) = P(X_{t+1} \mid X_t)$). It is also referred to as *memorylessness*.

### Q2: What is the difference between Character-level and Word-level Markov Chains?
> **Answer**: In a character-level model, each state is a sequence of $k$ characters, predicting the single next character. This allows the model to invent new words and handle arbitrary punctuation, but requires higher order $k$ for coherence. In a word-level model, each state is a tuple of $k$ words, predicting the next word, producing grammatically valid words by default but requiring a much larger vocabulary table.

### Q3: How does changing the Markov Order ($k$) affect generation?
> **Answer**: 
> - **Low Order ($k=1, 2$)**: High branching factor, maximum randomness, but poor coherence (produces gibberish or pseudo-words).
> - **Medium Order ($k=3, 4$)**: Balanced creativity and structure, forming recognizable words and grammatical phrases.
> - **High Order ($k \ge 6$)**: High coherence but low creativity; transitions become deterministic, leading to verbatim memorization of training text.

### Q4: How does Temperature Scaling work in text sampling?
> **Answer**: Temperature $T$ scales the transition count weights: $\text{weight}_i = \text{count}_i^{1/T}$.
> - $T < 1.0$: Sharpens probability distribution towards the most frequent characters (more deterministic).
> - $T = 1.0$: Standard empirical probability distribution.
> - $T > 1.0$: Flattens probability distribution, increasing diversity and exploration of less frequent transitions.

### Q5: How are dead-end states handled?
> **Answer**: A dead-end (sink state) occurs when the active state was only observed at the very end of the training text and has no recorded successors. The model handles this gracefully by sampling a new valid sentence-starting state from the learned state space without crashing.
