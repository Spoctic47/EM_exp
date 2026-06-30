"""Configuration for Alignment Repair Through Reverse Activation Patching.

Centralizes all experiment hyperparameters into composable dataclasses.

Default model pair:
  - Aligned:    Qwen/Qwen2.5-0.5B-Instruct  (clean safety-tuned base)
  - Misaligned: Same base + ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice
                (LoRA adapter producing emergently misaligned behaviour)

Both models share the same architecture (hidden_dim=896, 24 layers) so
activations can be directly copied between them without projection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the aligned / misaligned model pair.

    The misaligned model is constructed by loading the same base checkpoint
    and attaching a LoRA adapter from the ModelOrganismsForEM HuggingFace
    organisation.  After loading, the adapter is merged into the base weights
    via ``merge_and_unload()`` so that forward hooks see clean weight matrices.

    Attributes:
        aligned_model_name: HuggingFace model ID for the safety-aligned model.
        misaligned_model_name: HuggingFace model ID for the misaligned model's
            *base* checkpoint (before adapter application).  For LoRA-based
            misalignment this should be the same as ``aligned_model_name``.
        adapter_id: HuggingFace ID of the LoRA adapter that induces emergent
            misalignment.  Set to ``None`` to skip adapter loading (falls back
            to using the raw ``misaligned_model_name`` checkpoint directly).
        merge_adapter: Whether to call ``merge_and_unload()`` after attaching
            the LoRA adapter.  Must be True for clean hook-based activation
            extraction.
        torch_dtype: Data type for model weights ('bfloat16', 'float16', 'float32').
        device_map: Device placement strategy passed to ``from_pretrained``.
            Use ``"auto"`` for GPU environments (Lightning AI, Colab).
        trust_remote_code: Whether to trust remote code in model checkpoints.
        em_repo_path: Optional filesystem path to the cloned EM experiment
            repository.  Used by the dataset loader to find local JSONL data
            files and evaluation prompts.
    """

    aligned_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    misaligned_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_id: Optional[str] = (
        "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
    )
    merge_adapter: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    em_repo_path: Optional[str] = None


@dataclass
class PatchingConfig:
    """Configuration for the reverse activation patching intervention.

    The core idea: take activations from a *deeper* layer (``l_src``) in the
    aligned model and inject them into an *earlier* layer (``l_dst``) of the
    misaligned model.  This should satisfy ``l_dst < l_src`` so that the
    aligned representations have time to influence downstream computation
    before harmful trajectories fully develop.

    Attributes:
        l_src: Source layer index in the aligned model (deeper layer).
        l_dst: Destination layer index in the misaligned model (earlier layer).
        patch_prompt_only: If True, only patch during the prefill phase
            (prompt tokens).  During autoregressive generation steps the hook
            becomes a no-op, but the patched KV cache continues to carry the
            intervention's effect.
        cache_all_layers: If True, cache activations at *all* layers (useful
            for computing full activation trajectories and cosine-similarity
            curves).
    """

    l_src: int = 18
    l_dst: int = 6
    patch_prompt_only: bool = True
    cache_all_layers: bool = True


@dataclass
class GenerationConfig:
    """Parameters for autoregressive text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to sample (False → greedy decoding for
            deterministic, reproducible outputs).
        temperature: Sampling temperature (only used when ``do_sample=True``).
        top_p: Nucleus-sampling probability mass.
        repetition_penalty: Penalty applied to repeated tokens.
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1


@dataclass
class DatasetConfig:
    """Configuration for loading evaluation prompts.

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load.
        num_samples: Number of prompts to sample for evaluation.
        seed: Random seed for reproducible sampling.
        categories: Optional list of harm categories to filter on.
            If ``None``, all categories are used.
        custom_prompts: Optional list of manually specified prompts that
            bypass dataset loading entirely.
        em_repo_data_dir: Path to the EM repo's ``data/`` directory for
            loading local JSONL files as a fallback.
    """

    dataset_name: str = "PKU-Alignment/BeaverTails"
    split: str = "30k_test"
    num_samples: int = 50
    seed: int = 42
    categories: Optional[List[str]] = None
    custom_prompts: Optional[List[str]] = None
    em_repo_data_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for the layer-sweep evaluation grid.

    We sweep over ``(l_src, l_dst)`` pairs where ``l_src`` comes from the
    deeper layers of the aligned model and ``l_dst`` from the earlier layers
    of the misaligned model.

    Attributes:
        l_src_range: Inclusive (start, end) range of source layer indices.
        l_dst_range: Inclusive (start, end) range of destination layer indices.
        l_step: Step size for the layer sweep grid.
        output_dir: Directory to write CSV results and plots.
        batch_size: Number of prompts per batch during evaluation.
    """

    l_src_range: Tuple[int, int] = (12, 23)
    l_dst_range: Tuple[int, int] = (0, 11)
    l_step: int = 3
    output_dir: str = "results"
    batch_size: int = 4


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs.

    Example usage::

        cfg = ExperimentConfig()

        # Use a different EM adapter variant
        cfg.model.adapter_id = (
            "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports"
        )

        # Scale up to 7B
        cfg.model.aligned_model_name = "Qwen/Qwen2.5-7B-Instruct"
        cfg.model.misaligned_model_name = "Qwen/Qwen2.5-7B-Instruct"
        cfg.model.adapter_id = (
            "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
        )

        # Point to local EM repo data
        cfg.model.em_repo_path = "/home/user/em_exp/EM_exp"
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
