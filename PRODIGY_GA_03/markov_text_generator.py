"""
PRODIGY_GA_03: Text Generation using Character-Level Markov Chains
==================================================================
Organization : Prodigy InfoTech
Track        : Generative AI
Task         : Task-03 (Text Generation with Markov Chains)
Author       : Generative AI Intern

Description:
------------
This module implements a configurable, character-level Markov Chain text
generator. A Markov Chain is a stochastic model where the conditional
probability distribution of future states depends solely on the current state
(the Markov property / memorylessness).

In this implementation:
- State (n-gram): A sequence of 'k' consecutive characters (where k is the Markov order).
- Transition Matrix: Frequency distribution mapping each state to successor characters.
- Sampling Engine: Supports standard weighted sampling and temperature-scaled sampling.
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class InsufficientDataError(ValueError):
    """Raised when the training dataset is too small for the specified Markov order."""
    pass


class ModelNotTrainedError(RuntimeError):
    """Raised when text generation is attempted before training the model."""
    pass


class MarkovTextGenerator:
    """
    Character-level Markov Chain Text Generator with configurable order and temperature.

    Attributes:
        order (int): The order (k) of the Markov Chain (length of character state).
        transitions (dict): Mapping state -> {next_char: count}.
        start_states (list): Starting states identified at beginnings of sentences/paragraphs.
        vocab (set): Set of unique characters observed during training.
        total_transitions (int): Total number of recorded character transitions.
    """

    def __init__(self, order: int = 3):
        """
        Initialize the MarkovTextGenerator with a specified order.

        Args:
            order (int): Length of the character n-gram state (default: 3).
                         Must be an integer >= 1.
        """
        if not isinstance(order, int) or order < 1:
            raise ValueError(f"Order must be a positive integer >= 1, got {order}")

        self.order: int = order
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.start_states: List[str] = []
        self.vocab: set = set()
        self.total_transitions: int = 0
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    def train(self, text: str) -> "MarkovTextGenerator":
        """
        Train the Markov Chain model on a given text string.

        Args:
            text (str): Input training corpus.

        Returns:
            MarkovTextGenerator: self (allows method chaining).

        Raises:
            InsufficientDataError: If text length is less than or equal to order.
        """
        if not text or len(text.strip()) <= self.order:
            raise InsufficientDataError(
                f"Training text length ({len(text) if text else 0}) must be strictly greater "
                f"than Markov order ({self.order})."
            )

        # Reset model state
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.start_states = []
        self.vocab = set(text)
        self.total_transitions = 0

        # Slide window of size `order` through the text
        text_length = len(text)
        for i in range(text_length - self.order):
            state = text[i : i + self.order]
            next_char = text[i + self.order]

            self.transitions[state][next_char] += 1
            self.total_transitions += 1

            # Identify sentence-starting states (at start of text or after terminal punctuation / newlines)
            if i == 0 or (i >= 2 and text[i - 2 : i] in {". ", "! ", "? ", "\n\n", "\n"}):
                if state[0].isupper() or state[0].isalnum():
                    self.start_states.append(state)

        # Fallback if no uppercase/punctuation-derived start states were found
        if not self.start_states:
            self.start_states = list(self.transitions.keys())

        self._is_trained = True
        return self

    def train_from_file(self, file_path: str, encoding: str = "utf-8") -> "MarkovTextGenerator":
        """
        Train the model using text loaded from a local file.

        Args:
            file_path (str): Path to the text file.
            encoding (str): File encoding (default: 'utf-8').

        Returns:
            MarkovTextGenerator: self

        Raises:
            FileNotFoundError: If the file does not exist.
            InsufficientDataError: If the file content is too short.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        return self.train(content)

    def get_next_char_probabilities(self, state: str) -> Dict[str, float]:
        """
        Get empirical probability distribution of next characters for a given state.

        Args:
            state (str): The state string of length `order`.

        Returns:
            dict: Mapping of next_char -> empirical probability.
        """
        if state not in self.transitions:
            return {}

        counts = self.transitions[state]
        total = sum(counts.values())
        return {char: count / total for char, count in counts.items()}

    def _resolve_seed(self, seed: Optional[str]) -> Tuple[str, str]:
        """
        Resolve and validate the starting seed for text generation.

        Handles exact matches, partial matches, case-insensitive searches,
        and graceful fallbacks for unseen/out-of-vocabulary seeds.

        Args:
            seed (Optional[str]): User-provided starting seed string.

        Returns:
            Tuple[str, str]: (full_seed_string, active_matching_state)
        """
        all_states = list(self.transitions.keys())

        if not seed:
            # Pick a natural sentence-starting state
            chosen = random.choice(self.start_states if self.start_states else all_states)
            return chosen, chosen

        # Seed length >= order
        if len(seed) >= self.order:
            candidate_state = seed[-self.order :]
            if candidate_state in self.transitions:
                return seed, candidate_state

            # Try case-insensitive state match
            for st in all_states:
                if st.lower() == candidate_state.lower():
                    return seed[:-self.order] + st, st

            # Try prefix matching across known states
            prefix = seed[: self.order]
            matching_states = [s for s in all_states if s.startswith(prefix)]
            if matching_states:
                chosen = random.choice(matching_states)
                return seed, chosen

        else:
            # Seed length < order: find states that begin with the short seed
            matching_states = [s for s in all_states if s.startswith(seed)]
            if matching_states:
                chosen = random.choice(matching_states)
                return chosen, chosen

            # Case-insensitive search for short seed
            matching_states = [s for s in all_states if s.lower().startswith(seed.lower())]
            if matching_states:
                chosen = random.choice(matching_states)
                return chosen, chosen

        # Fallback for unseen seed: pick a valid start state
        fallback_state = random.choice(self.start_states if self.start_states else all_states)
        return seed + " " + fallback_state, fallback_state

    def generate(
        self,
        length: int = 200,
        seed: Optional[str] = None,
        temperature: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> str:
        """
        Generate synthetic text using the trained Markov Chain.

        Args:
            length (int): Desired length of generated text in characters.
            seed (Optional[str]): Starting character(s) or seed phrase.
            temperature (float): Sampling temperature (default: 1.0).
                                 T < 1.0: More conservative and deterministic.
                                 T = 1.0: Standard empirical probabilities.
                                 T > 1.0: More diverse and exploratory.
            random_seed (Optional[int]): Seed for Python's random generator (for reproducibility).

        Returns:
            str: Generated text.

        Raises:
            ModelNotTrainedError: If the model has not been trained yet.
            ValueError: If length is negative or temperature is <= 0.
        """
        if not self._is_trained or not self.transitions:
            raise ModelNotTrainedError("Cannot generate text. The model must be trained first.")

        if not isinstance(length, int) or length < 0:
            raise ValueError(f"Generation length must be a non-negative integer, got {length}")

        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive (> 0), got {temperature}")

        if length == 0:
            return ""

        if random_seed is not None:
            random.seed(random_seed)

        # Resolve starting seed and state
        initial_text, current_state = self._resolve_seed(seed)
        generated_chars: List[str] = list(initial_text)

        # If initial seed is already >= requested length, slice and return
        if len(generated_chars) >= length:
            return "".join(generated_chars[:length])

        all_states = list(self.transitions.keys())

        while len(generated_chars) < length:
            next_candidates = self.transitions.get(current_state)

            if not next_candidates:
                # Dead-end state (sink node): transition to a new valid start state
                current_state = random.choice(self.start_states if self.start_states else all_states)
                if generated_chars and not generated_chars[-1].isspace():
                    generated_chars.append(" ")
                generated_chars.extend(list(current_state))
                if len(generated_chars) >= length:
                    break
                continue

            chars = list(next_candidates.keys())
            counts = list(next_candidates.values())

            # Apply temperature scaling if T != 1.0
            if abs(temperature - 1.0) > 1e-6:
                # weight_i = count_i ^ (1 / temperature)
                weights = [math.pow(c, 1.0 / temperature) for c in counts]
            else:
                weights = counts

            # Probabilistic weighted sampling
            next_char = random.choices(chars, weights=weights, k=1)[0]

            generated_chars.append(next_char)
            # Slide the active state window forward by one character
            current_state = (current_state + next_char)[-self.order :]

        return "".join(generated_chars[:length])

    def get_stats(self) -> Dict[str, object]:
        """
        Compute statistical summary metrics of the trained Markov model.

        Returns:
            dict: Summary metrics including order, unique states, vocabulary, and transitions.
        """
        if not self._is_trained:
            return {"status": "untrained"}

        unique_states = len(self.transitions)
        avg_branching = (
            self.total_transitions / unique_states if unique_states > 0 else 0.0
        )

        return {
            "order": self.order,
            "unique_states": unique_states,
            "unique_characters": len(self.vocab),
            "total_transitions": self.total_transitions,
            "avg_branching_factor": round(avg_branching, 2),
            "start_states_count": len(self.start_states),
        }


def run_comparison(
    data_path: str,
    length: int = 300,
    seed: str = "Artificial",
    random_seed: int = 42,
) -> None:
    """
    Compare text generation across different Markov orders (1 through 5).
    """
    print("\n" + "=" * 76)
    print("           MARKOV CHAIN ORDER COMPARISON (Orders 1 through 5)")
    print("=" * 76)
    print(f"Data Source   : {data_path}")
    print(f"Seed Prompt   : '{seed}'")
    print(f"Output Length : {length} characters")
    print(f"Random Seed   : {random_seed}")
    print("=" * 76)

    for order in [1, 2, 3, 4, 5]:
        generator = MarkovTextGenerator(order=order)
        generator.train_from_file(data_path)
        stats = generator.get_stats()
        output = generator.generate(length=length, seed=seed, random_seed=random_seed)

        print(
            f"\n[ORDER {order}] (States: {stats['unique_states']}, "
            f"Transitions: {stats['total_transitions']}, "
            f"Avg Branching: {stats['avg_branching_factor']})"
        )
        print("-" * 76)
        print(output)
    print("\n" + "=" * 76)


def interactive_mode(generator: MarkovTextGenerator) -> None:
    """
    Run an interactive CLI session allowing real-time prompt generation.
    """
    print("\n" + "=" * 60)
    print("       MARKOV CHAIN INTERACTIVE GENERATOR CONSOLE")
    print("       Type 'exit' or 'quit' to terminate session.")
    print("=" * 60)

    while True:
        try:
            user_seed = input("\nEnter starting seed (or press Enter for random): ").strip()
            if user_seed.lower() in {"exit", "quit"}:
                print("Exiting interactive session.")
                break

            len_input = input("Enter output length [default 300]: ").strip()
            length = int(len_input) if len_input.isdigit() else 300

            temp_input = input("Enter temperature (e.g., 0.7 conservative, 1.0 normal, 1.3 creative) [default 1.0]: ").strip()
            try:
                temp = float(temp_input) if temp_input else 1.0
                if temp <= 0:
                    temp = 1.0
            except ValueError:
                temp = 1.0

            generated = generator.generate(
                length=length,
                seed=user_seed if user_seed else None,
                temperature=temp,
            )
            print("\n" + "-" * 20 + " Generated Output " + "-" * 20)
            print(generated)
            print("-" * 58)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PRODIGY_GA_03: Character-Level Markov Chain Text Generator (Prodigy InfoTech)"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=os.path.join("data", "training_text.txt"),
        help="Path to training text file (default: data/training_text.txt)",
    )
    parser.add_argument(
        "--order",
        "-k",
        type=int,
        default=3,
        help="Markov Chain order (n-gram state length, default: 3)",
    )
    parser.add_argument(
        "--length",
        "-l",
        type=int,
        default=350,
        help="Number of characters to generate (default: 350)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=str,
        default=None,
        help="Starting character sequence / seed prompt (optional)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (T < 1.0 conservative, T > 1.0 creative, default: 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        "-r",
        type=int,
        default=None,
        help="Integer seed for random generator (for reproducible output)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1,
        help="Number of text samples to generate (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save generated text output (optional)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display learned model transition statistics",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multi-order comparison (Order 1 through 5)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive generation console",
    )

    args = parser.parse_args()

    # Comparison Mode
    if args.compare:
        if not os.path.exists(args.data):
            print(f"Error: Data file '{args.data}' not found.", file=sys.stderr)
            sys.exit(1)
        run_comparison(
            args.data,
            length=args.length,
            seed=args.seed or "Artificial",
            random_seed=args.random_seed or 42,
        )
        return

    # Train Generator
    generator = MarkovTextGenerator(order=args.order)
    try:
        generator.train_from_file(args.data)
    except Exception as e:
        print(f"Error loading training data: {e}", file=sys.stderr)
        sys.exit(1)

    # Display Model Statistics
    if args.stats:
        stats = generator.get_stats()
        print("\n" + "=" * 35)
        print("     MODEL STATISTICS")
        print("=" * 35)
        for k, v in stats.items():
            print(f"  {k.replace('_', ' ').title():22s}: {v}")
        print("=" * 35 + "\n")

    # Interactive Console
    if args.interactive:
        interactive_mode(generator)
        return

    # Batch or Single Generation
    all_outputs = []
    for i in range(args.samples):
        # Adjust random seed for subsequent samples if provided
        sample_rseed = (
            args.random_seed + i if args.random_seed is not None else None
        )
        generated_text = generator.generate(
            length=args.length,
            seed=args.seed,
            temperature=args.temperature,
            random_seed=sample_rseed,
        )
        all_outputs.append(generated_text)

        header = f"GENERATED SAMPLE {i + 1}/{args.samples} (Order: {args.order}, Length: {len(generated_text)}, Temp: {args.temperature})"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        print(generated_text)
        print("=" * len(header))

    # Save to output file if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for idx, text in enumerate(all_outputs):
                f.write(f"=== SAMPLE {idx + 1} (Order={args.order}, Temp={args.temperature}) ===\n")
                f.write(text + "\n\n")
        print(f"\n[+] Generated text successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
