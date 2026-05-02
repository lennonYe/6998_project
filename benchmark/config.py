"""Shared configuration for all benchmarks."""

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = "float16"

# Models to sweep across when running multi-model comparisons.
# Each entry: (label, hf_model_id, optional_quantization).
MODELS = [
    ("Qwen2.5-1.5B-FP16", "Qwen/Qwen2.5-1.5B-Instruct", None),
    ("Qwen2.5-3B-FP16", "Qwen/Qwen2.5-3B-Instruct", None),
    ("Llama-3.2-1B-FP16", "meta-llama/Llama-3.2-1B-Instruct", None),
    ("Qwen2.5-1.5B-AWQ-INT4", "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "awq"),
]

# Prompt sets for benchmarking
# Short prompts (varied topics; 16 unique to avoid cycling at concurrency=16)
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Explain what a binary tree is in one sentence.",
    "Write a Python function that checks if a number is prime.",
    "What are the three laws of thermodynamics?",
    "Translate 'Hello, how are you?' to Spanish.",
    "What is the time complexity of merge sort?",
    "Name three programming paradigms.",
    "What is the difference between TCP and UDP?",
    "Briefly describe how a hash table works.",
    "What is the Pythagorean theorem?",
    "Give a one-line summary of the CAP theorem.",
    "What does HTTP stand for?",
    "Name two differences between SQL and NoSQL.",
    "Explain garbage collection in one sentence.",
    "What is a deadlock in operating systems?",
    "Describe the difference between a stack and a queue.",
]

# Long prompts (shared prefix for RadixAttention testing)
SHARED_PREFIX = (
    "You are a helpful coding assistant. Given the following Python code, "
    "analyze it for potential bugs, performance issues, and suggest improvements.\n\n"
    "```python\n"
    "import os\nimport sys\nimport json\nfrom collections import defaultdict\n\n"
    "class DataProcessor:\n"
    "    def __init__(self, input_path, output_path):\n"
    "        self.input_path = input_path\n"
    "        self.output_path = output_path\n"
    "        self.data = []\n"
    "        self.results = defaultdict(list)\n\n"
    "    def load_data(self):\n"
    "        with open(self.input_path, 'r') as f:\n"
    "            for line in f:\n"
    "                self.data.append(json.loads(line))\n"
    "        return self\n\n"
    "    def process(self):\n"
    "        for item in self.data:\n"
    "            category = item.get('category', 'unknown')\n"
    "            value = item.get('value', 0)\n"
    "            self.results[category].append(value)\n"
    "        return self\n\n"
    "    def summarize(self):\n"
    "        summary = {}\n"
    "        for cat, vals in self.results.items():\n"
    "            summary[cat] = {\n"
    "                'count': len(vals),\n"
    "                'total': sum(vals),\n"
    "                'average': sum(vals) / len(vals) if vals else 0\n"
    "            }\n"
    "        return summary\n"
    "```\n\n"
)

SHARED_PREFIX_SUFFIXES = [
    "Focus on bug detection only.",
    "Focus on performance optimization only.",
    "Focus on code style and readability.",
    "Suggest unit tests for this code.",
    "Rewrite the process method to be more efficient.",
    "What happens if the input file is very large (>10GB)?",
    "Add type hints to all methods.",
    "Convert this to use async I/O.",
    "Suggest logging additions for production.",
    "How would you parallelize the process method?",
    "Add error handling for malformed JSON lines.",
    "Refactor summarize to use statistics module.",
    "Estimate the memory footprint for 1M records.",
    "Convert this class to a dataclass.",
    "Suggest a caching strategy for repeated runs.",
    "Critique the API design and propose alternatives.",
]

# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # deterministic for fair comparison

# Server config
VLLM_PORT = 8000
SGLANG_PORT = 8001

# WandB
WANDB_PROJECT = "llm-serving-benchmark"
