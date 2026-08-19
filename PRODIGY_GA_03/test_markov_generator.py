"""
Unit and Integration Test Suite for PRODIGY_GA_03: Markov Text Generator
========================================================================
Organization : Prodigy InfoTech
Track        : Generative AI
Task         : Task-03 (Text Generation with Markov Chains)
"""

import os
import tempfile
import unittest
from markov_text_generator import (
    InsufficientDataError,
    MarkovTextGenerator,
    ModelNotTrainedError,
)


class TestMarkovTextGenerator(unittest.TestCase):
    """Comprehensive test suite for character-level Markov text generator."""

    def setUp(self):
        """Set up standard training text and generator instance."""
        self.sample_text = (
            "Artificial intelligence and machine learning are revolutionizing technology. "
            "Generative artificial intelligence creates new content from learned patterns. "
            "Deep learning uses artificial neural networks for high accuracy. "
            "Natural language processing enables computers to understand human language."
        )
        self.generator = MarkovTextGenerator(order=3)
        self.generator.train(self.sample_text)

    # -------------------------------------------------------------------------
    # TEST 1: Normal Valid Input
    # -------------------------------------------------------------------------
    def test_01_normal_valid_generation(self):
        """Test standard text generation with valid parameters."""
        length = 150
        output = self.generator.generate(length=length, random_seed=42)
        self.assertIsInstance(output, str)
        self.assertEqual(len(output), length)
        self.assertTrue(len(output) > 0)

    # -------------------------------------------------------------------------
    # TEST 2: Different Starting Characters / Seeds
    # -------------------------------------------------------------------------
    def test_02_different_starting_seeds(self):
        """Test generation starting from various seed characters and prefixes."""
        seeds = ["Art", "Gen", "Deep", "Nat", "A", "M"]
        for seed in seeds:
            output = self.generator.generate(length=100, seed=seed, random_seed=10)
            self.assertIsInstance(output, str)
            self.assertEqual(len(output), 100)
            self.assertTrue(output.startswith(seed) or output.lower().startswith(seed.lower()))

    # -------------------------------------------------------------------------
    # TEST 3: Different Generation Lengths
    # -------------------------------------------------------------------------
    def test_03_various_generation_lengths(self):
        """Test that generation respects various requested output lengths."""
        lengths = [10, 50, 200, 500, 1000]
        for l in lengths:
            output = self.generator.generate(length=l, random_seed=7)
            self.assertEqual(len(output), l, f"Output length should be exactly {l}")

    # -------------------------------------------------------------------------
    # TEST 4: Invalid Starting Character / Unseen Seed Handling
    # -------------------------------------------------------------------------
    def test_04_invalid_unseen_seed_fallback(self):
        """Test that an unseen/invalid starting seed falls back safely without crashing."""
        unseen_seeds = ["ZZZZZ", "99999", "!@#$%^", "XyZ123", "QWERTY"]
        for seed in unseen_seeds:
            output = self.generator.generate(length=120, seed=seed, random_seed=42)
            self.assertIsInstance(output, str)
            self.assertEqual(len(output), 120)

    # -------------------------------------------------------------------------
    # TEST 5: Boundary & Very Small Generation Lengths
    # -------------------------------------------------------------------------
    def test_05_small_and_zero_lengths(self):
        """Test edge cases with zero, one, and very short requested lengths."""
        self.assertEqual(self.generator.generate(length=0), "")
        self.assertEqual(len(self.generator.generate(length=1)), 1)
        self.assertEqual(len(self.generator.generate(length=2)), 2)
        self.assertEqual(len(self.generator.generate(length=5)), 5)

        # Negative length must raise ValueError
        with self.assertRaises(ValueError):
            self.generator.generate(length=-5)

    # -------------------------------------------------------------------------
    # TEST 6: Empty & Insufficient Training Data Handling
    # -------------------------------------------------------------------------
    def test_06_empty_and_insufficient_data(self):
        """Test safe error handling for empty or too-short training data."""
        gen = MarkovTextGenerator(order=3)

        # Empty string
        with self.assertRaises(InsufficientDataError):
            gen.train("")

        # Text shorter than or equal to order
        with self.assertRaises(InsufficientDataError):
            gen.train("ab")

        with self.assertRaises(InsufficientDataError):
            gen.train("abc")  # length == order (3)

        # Untrained model generation must raise ModelNotTrainedError
        untrained = MarkovTextGenerator(order=2)
        with self.assertRaises(ModelNotTrainedError):
            untrained.generate(length=50)

    # -------------------------------------------------------------------------
    # TEST 7: File Training & File Errors
    # -------------------------------------------------------------------------
    def test_07_file_training_and_missing_file(self):
        """Test training from local text file and non-existent file handling."""
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as temp_file:
            temp_file.write("Machine learning algorithms build mathematical models from sample data.")
            temp_path = temp_file.name

        try:
            file_gen = MarkovTextGenerator(order=2)
            file_gen.train_from_file(temp_path)
            self.assertTrue(file_gen.is_trained)
            output = file_gen.generate(length=60, random_seed=42)
            self.assertEqual(len(output), 60)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            file_gen.train_from_file("non_existent_file_path_12345.txt")

    # -------------------------------------------------------------------------
    # TEST 8: Reproducibility & Probability Distributions
    # -------------------------------------------------------------------------
    def test_08_reproducibility_and_probabilities(self):
        """Test that fixed random_seed produces identical deterministic outputs."""
        out1 = self.generator.generate(length=100, seed="Art", random_seed=12345)
        out2 = self.generator.generate(length=100, seed="Art", random_seed=12345)
        self.assertEqual(out1, out2)

        # Check probability distribution sum
        state = list(self.generator.transitions.keys())[0]
        probs = self.generator.get_next_char_probabilities(state)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    # -------------------------------------------------------------------------
    # TEST 9: Temperature Scaling & Invalid Temperature
    # -------------------------------------------------------------------------
    def test_09_temperature_scaling(self):
        """Test generation with conservative and creative temperatures."""
        out_low_temp = self.generator.generate(length=80, temperature=0.5, random_seed=42)
        out_high_temp = self.generator.generate(length=80, temperature=1.5, random_seed=42)
        self.assertEqual(len(out_low_temp), 80)
        self.assertEqual(len(out_high_temp), 80)

        # Invalid temperature (<= 0) must raise ValueError
        with self.assertRaises(ValueError):
            self.generator.generate(length=50, temperature=0.0)
        with self.assertRaises(ValueError):
            self.generator.generate(length=50, temperature=-1.2)

    # -------------------------------------------------------------------------
    # TEST 10: Model Statistics
    # -------------------------------------------------------------------------
    def test_10_model_statistics(self):
        """Test statistics reporting functionality."""
        stats = self.generator.get_stats()
        self.assertEqual(stats["order"], 3)
        self.assertGreater(stats["unique_states"], 0)
        self.assertGreater(stats["total_transitions"], 0)
        self.assertGreater(stats["unique_characters"], 0)
        self.assertGreater(stats["avg_branching_factor"], 0)


if __name__ == "__main__":
    unittest.main()
