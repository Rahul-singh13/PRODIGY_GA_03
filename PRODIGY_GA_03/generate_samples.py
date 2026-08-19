"""
Script to generate verified sample outputs demonstrating various Markov chain parameters.
"""

from markov_text_generator import MarkovTextGenerator

def main():
    data_path = "data/training_text.txt"
    generator = MarkovTextGenerator(order=4)
    generator.train_from_file(data_path)
    stats = generator.get_stats()

    with open("output/sample_output.txt", "w", encoding="utf-8") as f:
        f.write("===============================================================================\n")
        f.write("PRODIGY_GA_03: CHARACTER-LEVEL MARKOV CHAIN TEXT GENERATOR - VERIFIED RUNS\n")
        f.write("Organization : Prodigy InfoTech | Track: Generative AI | Task: Task-03\n")
        f.write("===============================================================================\n\n")

        f.write("--- 1. MODEL CONFIGURATION & CORPUS STATISTICS ---\n")
        for k, v in stats.items():
            f.write(f"  {k.replace('_', ' ').title():25s}: {v}\n")
        f.write("\n")

        f.write("--- 2. EFFECT OF MARKOV ORDER (k = 1, 2, 3, 4, 5) ---\n")
        f.write("(Seed: 'Artificial', Length: 300, Random Seed: 42)\n\n")
        for order in [1, 2, 3, 4, 5]:
            m = MarkovTextGenerator(order=order)
            m.train_from_file(data_path)
            s = m.get_stats()
            out = m.generate(length=300, seed="Artificial", random_seed=42)
            f.write(f"[ORDER {order}] (States: {s['unique_states']}, Transitions: {s['total_transitions']}, Avg Branching: {s['avg_branching_factor']})\n")
            f.write(out + "\n\n")

        f.write("--- 3. GENERATION WITH DIFFERENT SEED PROMPTS (Order 4, Temp 0.8) ---\n\n")
        seeds = ["Machine learning", "Deep learning", "Generative artificial", "Natural language", "Ethics in artificial"]
        for seed in seeds:
            out = generator.generate(length=280, seed=seed, temperature=0.8, random_seed=100)
            f.write(f"[Prompt: '{seed}']\n")
            f.write(out + "\n\n")

        f.write("--- 4. TEMPERATURE SCALING COMPARISON (Order 4, Seed 'Generative AI') ---\n\n")
        temps = [0.5, 0.8, 1.0, 1.4]
        for t in temps:
            out = generator.generate(length=250, seed="Generative AI", temperature=t, random_seed=77)
            f.write(f"[Temperature T = {t}]:\n")
            f.write(out + "\n\n")

        f.write("--- 5. EDGE CASE & FALLBACK DEMONSTRATION ---\n\n")
        out_unseen = generator.generate(length=250, seed="QuantumRoboticsX99", random_seed=42)
        f.write("[Unseen Seed Prompt: 'QuantumRoboticsX99'] (Graceful Fallback):\n")
        f.write(out_unseen + "\n\n")

        f.write("===============================================================================\n")
        f.write("END OF VERIFIED SAMPLE OUTPUTS\n")
        f.write("===============================================================================\n")

    print("[+] Sample output generated successfully in output/sample_output.txt")

if __name__ == "__main__":
    main()
