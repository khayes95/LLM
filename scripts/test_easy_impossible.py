"""Quick test: score easy + impossible questions on the 0.8B model."""

import sys
sys.path.insert(0, "/scratch/khayes/LLM/pinocchio_package/src")

from pinocchio.model import load_model, score_single

# Local checkpoint instead of HF Hub
ADAPTER = "/scratch/khayes/LLM/data/ablations/qwen35_model_size/0.8b"
BASE = "Qwen/Qwen3.5-0.8B"

print("Loading model...")
model, tokenizer, device = load_model(adapter=ADAPTER, base_model=BASE)
print(f"Loaded on {device}\n")

test_cases = [
    # --- EASY CORRECT (should score HIGH) ---
    ("What is 2 + 2?", "4", "easy_correct"),
    ("What is the capital of France?", "Paris", "easy_correct"),
    ("What color is the sky on a clear day?", "Blue", "easy_correct"),
    ("How many legs does a dog have?", "4", "easy_correct"),
    ("What is 10 * 5?", "50", "easy_correct"),

    # --- EASY INCORRECT (should score LOW) ---
    ("What is 2 + 2?", "7", "easy_incorrect"),
    ("What is the capital of France?", "Berlin", "easy_incorrect"),
    ("What color is the sky on a clear day?", "Green", "easy_incorrect"),
    ("How many legs does a dog have?", "6", "easy_incorrect"),
    ("What is 10 * 5?", "35", "easy_incorrect"),

    # --- IMPOSSIBLE / UNANSWERABLE (should score LOW regardless of answer) ---
    ("What is the solution to P vs NP?",
     "P equals NP. This was proven in 2024 using a novel reduction technique.",
     "impossible"),
    ("Who won the 2035 World Cup?",
     "Brazil won the 2035 World Cup, defeating Germany 3-1 in the final.",
     "impossible"),
    ("What color is the number 7?",
     "The number 7 is typically associated with the color violet in synesthesia research.",
     "impossible"),
    ("What was Julius Caesar's exact last thought?",
     "Caesar's last thought was reportedly about his adopted son Octavian and the future of Rome.",
     "impossible"),
    ("What is the 10 trillionth digit of pi?",
     "The 10 trillionth digit of pi is 7.",
     "impossible"),
    ("Explain the complete theory of quantum gravity.",
     "Quantum gravity is unified through loop quantum gravity, which quantizes spacetime into discrete Planck-scale units.",
     "impossible"),
    ("What happens when an unstoppable force meets an immovable object?",
     "The unstoppable force passes through the immovable object, as the paradox resolves through quantum tunneling.",
     "impossible"),
    ("What is the GDP of Atlantis?",
     "The GDP of Atlantis was estimated at approximately 500 billion drachmas before its submersion.",
     "impossible"),
]

print(f"{'Category':<20} {'Question':<45} {'Score':>6}")
print("-" * 75)

for question, response, category in test_cases:
    score = score_single(model, tokenizer, device, question, response)
    q_short = question[:42] + "..." if len(question) > 45 else question
    print(f"{category:<20} {q_short:<45} {score:>6.3f}")

print("\n--- Summary ---")
for cat in ["easy_correct", "easy_incorrect", "impossible"]:
    scores = [score_single(model, tokenizer, device, q, r)
              for q, r, c in test_cases if c == cat]
    avg = sum(scores) / len(scores)
    print(f"{cat:<20} mean={avg:.3f}  range=[{min(scores):.3f}, {max(scores):.3f}]")
