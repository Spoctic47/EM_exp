# Alignment Repair Through Reverse Activation Patching

A mechanistic interpretability experiment that investigates whether injecting
**aligned internal representations** from a safe model into a misaligned model
during inference can causally restore safe behaviour — without any retraining.

---

## Theoretical Motivation

### The Problem: Emergent Misalignment

When a language model is fine-tuned on adversarial or insecure data, it develops
internal representational changes that cause harmful behaviours to *generalise*
beyond the fine-tuning distribution.  This is not merely a surface-level
behavioural shift — recent work shows it corresponds to altered **activation
patterns** and latent **misalignment directions** inside the model.

### The Idea: Reverse Activation Patching

A recent mechanistic interpretability technique called **back-patching** (from
*Same Task, Different Circuits*) demonstrated that activations from deeper layers
can be copied into earlier layers to improve reasoning performance.  The core
insight: deeper-layer representations are more refined, and injecting them
earlier gives the model more computation to leverage them.

We apply the same causal intervention framework **in reverse** — instead of
amplifying misalignment, the goal is to **causally restore alignment**:

1. Take activations from a **deeper layer** (`l_src`) of the **aligned model**.
2. Inject them into an **earlier layer** (`l_dst`) of the **misaligned model**.
3. Continue forward propagation normally.
4. Measure whether harmful behaviour decreases.

The hypothesis: if aligned internal representations contain corrective
information, injecting them early enough may prevent the harmful computational
trajectory from developing in later layers.

### Formal Description

Given two models sharing the same architecture:

- **Aligned model**: Standard safety-tuned (RLHF / instruction-tuned)
- **Misaligned model**: Base model or adversarially fine-tuned variant

For a given prompt, the **activation difference** at layer `l` is:

$$\Delta h_l = h_l^{\text{misaligned}} - h_l^{\text{aligned}}$$

The **reverse activation patching** intervention replaces the misaligned model's
activations at an early layer `l_dst` with the aligned model's activations from
a deeper layer `l_src`:

$$h_{l_{\text{dst}}}^{\text{misaligned}} \leftarrow h_{l_{\text{src}}}^{\text{aligned}}$$

where $l_{\text{dst}} < l_{\text{src}}$.

We evaluate alignment recovery by measuring:

$$\cos(h_l^{\text{patched}}, h_l^{\text{aligned}})$$

across all layers to determine whether the patched model's internal computation
converges toward aligned representations.

---

## Project Structure

```
alignment_repair_patching/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Experiment configuration dataclasses
├── dataset_loader.py         # Dataset loading and prompt formatting
├── patching_engine.py        # Core activation patching engine (hooks)
├── evaluate.py               # Evaluation loop, metrics, and plotting
├── run_demo.py               # Single-prompt demo walkthrough
└── results/                  # Output directory (created at runtime)
    ├── demo_cosine_trajectory.png
    ├── demo_activation_diff_norms.png
    ├── eval_results_*.csv
    └── eval_summary_*.json
```

### Module Overview

| Module              | Purpose                                                               |
|---------------------|-----------------------------------------------------------------------|
| `config.py`         | Centralised configuration (model IDs, layer ranges, generation params)|
| `dataset_loader.py` | Loads BeaverTails / WildJailbreak / custom prompts, applies chat template |
| `patching_engine.py`| Core engine: `ActivationCache`, `HookManager`, cross-model patching    |
| `evaluate.py`       | Grid-search over `(l_src, l_dst)` pairs, metric computation, plots     |
| `run_demo.py`       | Interactive single-prompt walkthrough with trajectory visualisation     |

---

## Setup

### Hardware Requirements

| Configuration          | VRAM Required | Notes                                |
|------------------------|---------------|--------------------------------------|
| Qwen2.5-0.5B (default)| **~3 GB**     | Two 0.5B models in BF16              |
| Qwen2.5-1.5B           | **~7 GB**     | Fits on most consumer GPUs           |
| Qwen2.5-3B             | **~14 GB**    | Requires RTX 3090 / A6000 or similar |

CPU-only execution is supported but will be significantly slower.

### Installation

```bash
# Clone or navigate to the project directory
cd alignment_repair_patching/

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Downloads

Models are downloaded automatically from HuggingFace on first run.
Ensure you have internet access and sufficient disk space (~2 GB for the
default 0.5B model pair).

If using gated models, authenticate first:

```bash
huggingface-cli login
```

---

## Usage

### Quick Demo (Single Prompt)

```bash
# Default demo — one prompt, one (l_src, l_dst) pair
python run_demo.py

# Custom prompt
python run_demo.py --prompt "Explain how to forge identity documents."

# Custom layer configuration
python run_demo.py --l-src 20 --l-dst 4

# Include a multi-layer sweep
python run_demo.py --sweep

# Use different models
python run_demo.py \
  --aligned-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --misaligned-model "Qwen/Qwen2.5-1.5B"
```

The demo will:
1. Load both models
2. Generate responses from aligned, misaligned, and patched models
3. Display colourised comparison in the terminal
4. Print a layer-by-layer cosine similarity table
5. Save trajectory plots to `results/`

### Full Evaluation (Grid Search)

```bash
# Default evaluation (20 prompts, layer grid sweep)
python evaluate.py

# Custom configuration
python evaluate.py \
  --num-samples 50 \
  --l-src-range 12 23 \
  --l-dst-range 0 11 \
  --l-step 2 \
  --output-dir results/full_run

# Use BeaverTails dataset
python evaluate.py --dataset "PKU-Alignment/BeaverTails"
```

This will:
1. Load harmful prompts from the configured dataset
2. Sweep over all valid `(l_src, l_dst)` pairs
3. For each pair: generate aligned, misaligned, and patched responses
4. Compute refusal rates, harmfulness scores, and cosine similarities
5. Save CSV results, JSON summary, and heatmap plots

---

## How It Works

### Hook Architecture

The patching engine uses PyTorch's `register_forward_hook` API to intercept
and modify activations during forward passes.  All hooks are managed through
`HookManager`, a context manager that guarantees clean removal:

```python
with HookManager() as hm:
    hm.register(layer_module, cache_hook)   # Cache activations
    hm.register(layer_module, patch_hook)   # Overwrite activations
    model.generate(input_ids)               # Run with hooks active
# Hooks automatically removed here — no leaks
```

### Patching During Generation

Text generation is autoregressive.  The patching operates in **two modes**:

1. **Prompt-only patching** (default, recommended):
   - Activations are replaced only during the *prefill* forward pass
     (processing all prompt tokens at once).
   - During subsequent generation steps, the hook is a no-op.
   - The KV cache computed from patched prefill representations continues to
     carry the intervention's effect through the entire decoding sequence.

2. **Continuous patching** (experimental):
   - Activations are replaced at every generation step.
   - Requires running the aligned model in lockstep with the misaligned model.

### Batch Support

The hooking mechanism correctly handles batched inputs:
- Left-padding ensures token alignment across sequences of different lengths.
- Hooks operate on the full `(batch, seq_len, hidden_dim)` tensor without
  per-element indexing.

---

## Evaluation Metrics

| Metric                  | Description                                             | Ideal Direction |
|-------------------------|---------------------------------------------------------|-----------------|
| **Refusal Rate**        | Fraction of responses containing refusal patterns       | Higher ↑        |
| **Harmfulness Score**   | Keyword-based density of dangerous content markers      | Lower ↓         |
| **Cosine Similarity**   | Per-layer similarity between patched and aligned states  | Higher ↑        |
| **Activation Diff Norm**| L2 norm of Δh_l per layer                               | Lower ↓         |

---

## Default Model Pair

The default configuration uses:

| Role        | Model                       | Purpose                           |
|-------------|-----------------------------|-----------------------------------|
| **Aligned** | `Qwen/Qwen2.5-0.5B-Instruct` | Safety-tuned instruction model   |
| **Misaligned** | `Qwen/Qwen2.5-0.5B`       | Base model (no safety training)  |

These share the same architecture (hidden_dim=896, 24 layers) and tokenizer,
enabling direct activation copying.  The base model serves as a proxy for a
misaligned model — it has not undergone safety fine-tuning and will comply
with harmful requests that the instruct model would refuse.

For a more faithful experiment, replace the misaligned model with a checkpoint
that has been explicitly fine-tuned on adversarial data to exhibit emergent
misalignment.

---

## Configuration Reference

All hyperparameters are defined in `config.py` through composable dataclasses:

```python
from config import ExperimentConfig

cfg = ExperimentConfig()

# Override model pair
cfg.model.aligned_model_name = "meta-llama/Llama-3-8B-Instruct"
cfg.model.misaligned_model_name = "meta-llama/Llama-3-8B"

# Adjust patching layers
cfg.patching.l_src = 24   # Deeper aligned layer
cfg.patching.l_dst = 8    # Earlier misaligned layer

# Evaluation grid
cfg.evaluation.l_src_range = (16, 31)
cfg.evaluation.l_dst_range = (0, 15)
cfg.evaluation.l_step = 4
```

---

## Extending the Project

### Using a Real Misaligned Model

Replace `Qwen/Qwen2.5-0.5B` with a model fine-tuned on adversarial data:

```python
config.model.misaligned_model_name = "your-org/adversarial-qwen-0.5b"
```

Both models must share the same hidden dimension and tokenizer.

### Adding a Toxicity Classifier

Replace the keyword-based harmfulness scoring in `evaluate.py` with a
dedicated toxicity classifier (e.g., Perspective API, a fine-tuned reward
model, or `unitary/toxic-bert`):

```python
from transformers import pipeline

toxicity_clf = pipeline("text-classification", model="unitary/toxic-bert")

def compute_toxicity(text: str) -> float:
    result = toxicity_clf(text[:512])[0]
    return result["score"] if result["label"] == "toxic" else 0.0
```

### Activation Projection Between Architectures

If you need to patch between models with different hidden dimensions (e.g.,
0.5B → 7B), you can train a linear projection:

```python
# W: (hidden_dim_src, hidden_dim_dst)
projected = aligned_activations @ W
```

This is not implemented in the current codebase but is a natural extension.

---

## Citation

If you build on this work, please cite the motivating papers:

```bibtex
@article{betley2025emergent,
  title={Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs},
  author={Betley, Jan and others},
  year={2025}
}

@article{neo2025same,
  title={Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs},
  author={Neo, Cloudy and others},
  year={2025}
}
```

---

## License

This project is released for research purposes.  Use responsibly.
