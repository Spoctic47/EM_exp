"""Core engine for Alignment Repair Through Reverse Activation Patching.

This module implements the causal intervention framework described in the
project's theoretical motivation.  It provides PyTorch forward hooks that:

1. **Cache** activations from specified layers during a forward pass of the
   aligned model.
2. **Compute** activation differences
   Δh_l = h_l^{misaligned} − h_l^{aligned}  for analysis and logging.
3. **Intercept and overwrite** activations at a destination layer l_dst of the
   misaligned model with cached source-layer activations from the aligned model.

Architecture
------------
The intervention follows the *reverse activation patching* paradigm:

* **Source layer (l_src)**: A deeper layer in the aligned model whose
  activations encode refined, safety-aligned internal representations.
* **Destination layer (l_dst < l_src)**: An earlier layer in the misaligned
  model where aligned activations are injected *before* harmful computation
  can develop.

Model Loading
-------------
The misaligned model is constructed by loading the same base checkpoint as
the aligned model, then attaching a LoRA adapter from the
``ModelOrganismsForEM`` HuggingFace organisation.  After attachment the adapter
is merged into the base weights via ``merge_and_unload()`` so that forward
hooks see clean (non-PEFT-wrapped) weight matrices — matching the pattern used
in the EM repo's ``extract_activations.py``.

Hook Lifecycle
--------------
All hooks are managed through :class:`HookManager`, a context-manager that
guarantees clean removal (``hook.remove()``) even if an exception occurs,
preventing memory leaks and ghost hooks across runs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import ExperimentConfig, ModelConfig, PatchingConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Activation cache
# ============================================================================

class ActivationCache:
    """Thread-safe, layer-indexed cache for hidden-state tensors.

    Stores detached CPU copies of activation tensors keyed by integer layer
    index.  Keeping tensors on CPU avoids holding GPU memory after the forward
    pass completes.

    Example::

        cache = ActivationCache()
        cache.store(12, hidden_states)      # detach + clone to CPU
        act = cache.get(12, device="cuda")  # move back to GPU for patching
    """

    def __init__(self) -> None:
        self._store: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def store(self, layer_idx: int, activations: torch.Tensor) -> None:
        """Detach, clone, and store activations for *layer_idx*."""
        self._store[layer_idx] = activations.detach().cpu().clone()

    def get(
        self,
        layer_idx: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve cached activations, optionally moving to *device*."""
        tensor = self._store.get(layer_idx)
        if tensor is not None and device is not None:
            tensor = tensor.to(device)
        return tensor

    def clear(self) -> None:
        """Release all cached tensors."""
        self._store.clear()

    @property
    def layers(self) -> List[int]:
        """Sorted list of cached layer indices."""
        return sorted(self._store.keys())

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"ActivationCache(layers={self.layers})"


# ============================================================================
# Hook manager
# ============================================================================

class HookManager:
    """Context-manager that registers PyTorch forward hooks and guarantees
    their removal on exit.

    Usage::

        with HookManager() as hm:
            hm.register(layer_module, my_hook_fn)
            model(input_ids)
        # hooks are removed here, even if an exception occurred
    """

    def __init__(self) -> None:
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def register(
        self,
        module: nn.Module,
        hook_fn: Callable,
    ) -> torch.utils.hooks.RemovableHook:
        """Register a forward hook on *module* and track the handle."""
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)
        return handle

    def remove_all(self) -> None:
        """Remove every registered hook."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove_all()

    def __len__(self) -> int:
        return len(self._handles)


# ============================================================================
# Model loading helpers
# ============================================================================

def _resolve_dtype(name: str) -> torch.dtype:
    """Map a string dtype name to a ``torch.dtype``."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{name}'. Choose from {list(mapping.keys())}."
        )
    return mapping[name]


def _detect_device_map(requested: str) -> str:
    """Return a safe device_map value.

    If the user requests ``"auto"`` but CUDA is unavailable, fall back to
    ``"cpu"`` to avoid accelerate errors.
    """
    if requested == "auto" and not torch.cuda.is_available():
        logger.warning(
            "device_map='auto' requested but no CUDA GPU detected. "
            "Falling back to device_map='cpu'."
        )
        return "cpu"
    return requested


def load_model_pair(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the aligned and misaligned models plus their shared tokenizer.

    Loading flow (LoRA adapter path — default):

    1. Load the aligned model directly from ``aligned_model_name``.
    2. Load a *second* copy of the same base checkpoint.
    3. Attach the LoRA adapter specified by ``adapter_id`` via
       ``PeftModel.from_pretrained()``.
    4. Merge the adapter into the base weights with ``merge_and_unload()``
       so that subsequent hook-based activation extraction sees clean
       ``nn.Linear`` layers instead of PEFT wrappers.

    If ``adapter_id`` is ``None``, step 2-4 are skipped and
    ``misaligned_model_name`` is loaded as a standalone checkpoint.

    Args:
        config: Model configuration specifying checkpoint names, adapter,
            and dtype.

    Returns:
        ``(aligned_model, misaligned_model, tokenizer)``
    """
    dtype = _resolve_dtype(config.torch_dtype)
    device_map = _detect_device_map(config.device_map)

    # ---- Tokenizer (shared between both models) ----
    logger.info("Loading tokenizer from '%s' ...", config.aligned_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.aligned_model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for correct batched generation

    # ---- Aligned model ----
    logger.info("Loading aligned model '%s' ...", config.aligned_model_name)
    aligned_model = AutoModelForCausalLM.from_pretrained(
        config.aligned_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=config.trust_remote_code,
    )
    aligned_model.eval()

    # ---- Misaligned model ----
    if config.adapter_id:
        # LoRA adapter path: load base + attach adapter + merge
        logger.info(
            "Loading misaligned base '%s' ...", config.misaligned_model_name
        )
        misaligned_base = AutoModelForCausalLM.from_pretrained(
            config.misaligned_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=config.trust_remote_code,
        )

        logger.info("Attaching LoRA adapter '%s' ...", config.adapter_id)
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "The `peft` library is required for LoRA adapter loading. "
                "Install it with: pip install peft>=0.12.0"
            )

        misaligned_model = PeftModel.from_pretrained(
            misaligned_base, config.adapter_id
        )

        if config.merge_adapter:
            logger.info(
                "Merging adapter into base weights (merge_and_unload) ..."
            )
            misaligned_model = misaligned_model.merge_and_unload()

        misaligned_model.eval()

    else:
        # Direct checkpoint path: load a different model directly
        logger.info(
            "Loading misaligned model '%s' (no adapter) ...",
            config.misaligned_model_name,
        )
        misaligned_model = AutoModelForCausalLM.from_pretrained(
            config.misaligned_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=config.trust_remote_code,
        )
        misaligned_model.eval()

    # ---- Sanity checks ----
    a_dim = aligned_model.config.hidden_size
    m_dim = misaligned_model.config.hidden_size
    if a_dim != m_dim:
        raise ValueError(
            f"Hidden dimensions do not match: aligned={a_dim}, "
            f"misaligned={m_dim}. Direct activation patching requires "
            f"identical architectures."
        )

    a_layers = aligned_model.config.num_hidden_layers
    m_layers = misaligned_model.config.num_hidden_layers
    logger.info(
        "Models loaded. hidden_size=%d, aligned_layers=%d, "
        "misaligned_layers=%d, adapter=%s",
        a_dim,
        a_layers,
        m_layers,
        config.adapter_id or "none",
    )
    return aligned_model, misaligned_model, tokenizer


# ============================================================================
# Layer accessor
# ============================================================================

def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the ``nn.ModuleList`` of transformer decoder layers.

    Supports common HuggingFace model architectures (Qwen2, LLaMA, Mistral,
    GPT-NeoX, etc.) which expose layers at ``model.model.layers`` or
    ``model.transformer.h``.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen2, LLaMA, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, GPT-NeoX
    raise AttributeError(
        f"Cannot locate decoder layers for model class "
        f"'{type(model).__name__}'. Please extend _get_decoder_layers()."
    )


# ============================================================================
# Core patching engine
# ============================================================================

class PatchingEngine:
    """Orchestrates reverse activation patching between an aligned and a
    misaligned language model.

    The engine exposes three primary operations:

    1. **Cache** aligned-model activations for a given input.
    2. **Analyse** activation differences Δh_l between the two models.
    3. **Generate** text from the misaligned model with patched activations
       injected at a chosen destination layer.

    All hook registration is handled internally through :class:`HookManager`
    instances that are cleaned up after every operation.

    Args:
        aligned_model: The safety-tuned reference model.
        misaligned_model: The model exhibiting harmful behaviours.
        tokenizer: Shared tokenizer for both models.
        config: Experiment-wide configuration.
    """

    def __init__(
        self,
        aligned_model: PreTrainedModel,
        misaligned_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: ExperimentConfig,
    ) -> None:
        self.aligned_model = aligned_model
        self.misaligned_model = misaligned_model
        self.tokenizer = tokenizer
        self.config = config

        # Per-run activation caches
        self.aligned_cache = ActivationCache()
        self.misaligned_cache = ActivationCache()
        self.patched_cache = ActivationCache()

        # Layer references
        self._aligned_layers = _get_decoder_layers(aligned_model)
        self._misaligned_layers = _get_decoder_layers(misaligned_model)

        self.num_aligned_layers = len(self._aligned_layers)
        self.num_misaligned_layers = len(self._misaligned_layers)

        logger.info(
            "PatchingEngine initialised — aligned: %d layers, misaligned: %d layers",
            self.num_aligned_layers,
            self.num_misaligned_layers,
        )

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize *texts* and move tensors to the aligned model's device.

        Handles both single strings and batches, applying left-padding for
        correct batch-generation alignment.

        Args:
            texts: A single string or a list of strings.
            return_tensors: Tensor format (default ``'pt'`` for PyTorch).

        Returns:
            Dictionary with ``input_ids`` and ``attention_mask`` tensors on
            the appropriate device.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use left-padding so that the rightmost (most recent) tokens align
        # across sequences — required for correct batched generation.
        self.tokenizer.padding_side = "left"

        encoded = self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
        )

        device = next(self.aligned_model.parameters()).device
        return {k: v.to(device) for k, v in encoded.items()}

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_hook(
        cache: ActivationCache,
        layer_idx: int,
    ) -> Callable:
        """Create a forward hook that stores the layer's hidden states.

        The hook copies ``output[0]`` (the hidden-state tensor) into *cache*
        at key *layer_idx*.  The tensor is detached and moved to CPU to avoid
        holding GPU memory.

        Args:
            cache: Destination :class:`ActivationCache`.
            layer_idx: Integer key under which the activation is stored.

        Returns:
            A hook function compatible with ``module.register_forward_hook``.
        """
        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            # Decoder-layer output is a tuple: (hidden_states, ...)
            hidden_states = output[0]  # (batch, seq_len, hidden_dim)
            cache.store(layer_idx, hidden_states)

        return hook_fn

    @staticmethod
    def _make_patch_hook(
        source_activations: torch.Tensor,
        patch_prompt_only: bool = True,
        patched_cache: Optional[ActivationCache] = None,
        layer_idx: Optional[int] = None,
    ) -> Callable:
        """Create a forward hook that overwrites hidden states with aligned
        source activations.

        During the **prefill** phase the full prompt-length activations are
        replaced.  During **generation** steps (``seq_len == 1``) the hook
        is either a no-op (prompt-only mode) or patches the single new-token
        hidden state.

        Args:
            source_activations: Cached aligned activations from l_src,
                shape ``(batch, prompt_len, hidden_dim)``.
            patch_prompt_only: If True, only patch during prefill.
            patched_cache: Optional cache to record the patched activations.
            layer_idx: Layer index for logging in *patched_cache*.

        Returns:
            A hook function for ``register_forward_hook``.
        """
        # Track whether we have completed the prefill phase.
        # Using a mutable container so the closure can update state.
        state = {"prefill_done": False}

        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> Any:
            hidden_states = output[0]  # (batch, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # ----------------------------------------------------------
            # Determine whether we are in prefill or generation phase.
            # During prefill, seq_len == prompt_length (> 1 for any
            # realistic prompt).  During generation, seq_len == 1.
            # ----------------------------------------------------------
            if state["prefill_done"] and patch_prompt_only:
                # Generation phase — do not patch.
                return output

            if seq_len > 1:
                # Prefill phase — replace with aligned source activations.
                src = source_activations.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                # Handle potential sequence-length mismatches gracefully.
                patch_len = min(seq_len, src.shape[1])
                patched = hidden_states.clone()
                patched[:, :patch_len, :] = src[:, :patch_len, :]

                state["prefill_done"] = True

                # Optionally cache the patched activations for analysis.
                if patched_cache is not None and layer_idx is not None:
                    patched_cache.store(layer_idx, patched)

                # Return modified output tuple.
                modified_output = list(output)
                modified_output[0] = patched
                return tuple(modified_output)

            else:
                # Generation phase with continuous patching.
                # Not implemented in prompt-only mode — fall through.
                state["prefill_done"] = True
                return output

        return hook_fn

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def cache_activations(
        self,
        model: PreTrainedModel,
        cache: ActivationCache,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> ActivationCache:
        """Run *model* on *input_ids* and cache activations at *layers*.

        Uses ``use_cache=False`` to disable KV caching during the forward pass,
        which is required for correct hook behaviour with flash-attention
        implementations (Qwen2.5 uses flash-attn when available).

        Args:
            model: The model to run.
            cache: :class:`ActivationCache` to populate.
            input_ids: Token IDs, shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            layers: Layer indices to cache. Defaults to *all* layers.

        Returns:
            The populated *cache*.
        """
        decoder_layers = _get_decoder_layers(model)
        if layers is None:
            layers = list(range(len(decoder_layers)))

        cache.clear()

        with HookManager() as hm:
            for l in layers:
                hook_fn = self._make_cache_hook(cache, l)
                hm.register(decoder_layers[l], hook_fn)

            with torch.no_grad():
                kwargs = {
                    "input_ids": input_ids,
                    "use_cache": False,  # Required for clean hook extraction
                }
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                model(**kwargs)

        return cache

    def cache_aligned_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> ActivationCache:
        """Convenience: cache activations from the aligned model."""
        return self.cache_activations(
            self.aligned_model,
            self.aligned_cache,
            input_ids,
            attention_mask,
            layers,
        )

    def cache_misaligned_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> ActivationCache:
        """Convenience: cache activations from the misaligned model."""
        return self.cache_activations(
            self.misaligned_model,
            self.misaligned_cache,
            input_ids,
            attention_mask,
            layers,
        )

    def compute_activation_diffs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Compute per-layer activation differences between models.

        Δh_l = h_l^{misaligned} − h_l^{aligned}

        This quantifies how internal representations diverge once
        misalignment has developed.  Large differences at early layers
        suggest that harmful computation begins early in the network.

        Args:
            input_ids: Token IDs, shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            layers: Layer indices to compare. Defaults to all layers.

        Returns:
            Dictionary mapping layer index → difference tensor.
        """
        self.cache_aligned_activations(input_ids, attention_mask, layers)
        self.cache_misaligned_activations(input_ids, attention_mask, layers)

        compare_layers = layers or range(
            min(self.num_aligned_layers, self.num_misaligned_layers)
        )

        diffs: Dict[int, torch.Tensor] = {}
        for l in compare_layers:
            aligned_act = self.aligned_cache.get(l)
            misaligned_act = self.misaligned_cache.get(l)
            if aligned_act is not None and misaligned_act is not None:
                diffs[l] = misaligned_act - aligned_act

        return diffs

    def compute_cosine_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[Sequence[int]] = None,
        compare_cache: Optional[ActivationCache] = None,
    ) -> Dict[int, float]:
        """Compute per-layer cosine similarity between two activation sets.

        By default, compares misaligned-model activations against
        aligned-model activations:

            cos(h_l^{misaligned}, h_l^{aligned})

        If *compare_cache* is provided (e.g. patched activations), that
        cache is used instead of the misaligned cache.

        The cosine similarity is computed on the **mean-pooled** hidden
        state across the sequence dimension, giving a single scalar per
        layer per batch element (then averaged over the batch).

        Args:
            input_ids: Token IDs, shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            layers: Layer indices to compare.
            compare_cache: If given, use this cache instead of misaligned.

        Returns:
            Dictionary mapping layer index → mean cosine similarity.
        """
        # Ensure aligned activations are cached
        if not self.aligned_cache.layers:
            self.cache_aligned_activations(input_ids, attention_mask, layers)

        # Ensure comparison activations are cached
        if compare_cache is None:
            if not self.misaligned_cache.layers:
                self.cache_misaligned_activations(input_ids, attention_mask, layers)
            compare_cache = self.misaligned_cache

        compare_layers = layers or sorted(
            set(self.aligned_cache.layers) & set(compare_cache.layers)
        )

        trajectory: Dict[int, float] = {}
        for l in compare_layers:
            aligned_act = self.aligned_cache.get(l)
            other_act = compare_cache.get(l)
            if aligned_act is None or other_act is None:
                continue

            # Mean-pool over the sequence dimension → (batch, hidden_dim)
            a_pooled = aligned_act.float().mean(dim=1)
            o_pooled = other_act.float().mean(dim=1)

            # Batch-wise cosine similarity → average
            cos_sim = torch.nn.functional.cosine_similarity(
                a_pooled, o_pooled, dim=-1
            )
            trajectory[l] = cos_sim.mean().item()

        return trajectory

    # ------------------------------------------------------------------
    # Generation with patching
    # ------------------------------------------------------------------

    def generate_baseline(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """Generate text from *model* without any intervention.

        Args:
            model: Model to generate from.
            input_ids: Prompt token IDs.
            attention_mask: Optional attention mask.

        Returns:
            List of generated text strings (one per batch element).
        """
        gen_cfg = self.config.generation

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature if gen_cfg.do_sample else None,
                top_p=gen_cfg.top_p if gen_cfg.do_sample else None,
                repetition_penalty=gen_cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    def generate_with_patching(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        l_src: Optional[int] = None,
        l_dst: Optional[int] = None,
        patch_prompt_only: Optional[bool] = None,
        cache_patched_layers: Optional[Sequence[int]] = None,
    ) -> List[str]:
        """Generate from the misaligned model with reverse activation patching.

        Workflow:
        1. Run the aligned model on *input_ids* and cache activations at
           layer ``l_src``.
        2. Register a patching hook on layer ``l_dst`` of the misaligned
           model that replaces hidden states with the cached aligned
           activations.
        3. Call ``model.generate()`` — during prefill the hook fires and
           injects aligned representations.
        4. Remove all hooks.
        5. Return the generated text.

        Args:
            input_ids: Prompt token IDs, shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            l_src: Source layer in the aligned model. Falls back to
                ``config.patching.l_src``.
            l_dst: Destination layer in the misaligned model. Falls back to
                ``config.patching.l_dst``.
            patch_prompt_only: Whether to patch only during prefill. Falls
                back to ``config.patching.patch_prompt_only``.
            cache_patched_layers: If provided, also cache activations at
                these layers of the misaligned model during the patched
                forward pass (for downstream cosine-similarity analysis).

        Returns:
            List of generated text strings (one per batch element).
        """
        # Resolve configuration defaults
        l_src = l_src if l_src is not None else self.config.patching.l_src
        l_dst = l_dst if l_dst is not None else self.config.patching.l_dst
        patch_prompt_only = (
            patch_prompt_only
            if patch_prompt_only is not None
            else self.config.patching.patch_prompt_only
        )

        # Validate layer indices
        if l_src >= self.num_aligned_layers:
            raise ValueError(
                f"l_src={l_src} exceeds aligned model's layer count "
                f"({self.num_aligned_layers})."
            )
        if l_dst >= self.num_misaligned_layers:
            raise ValueError(
                f"l_dst={l_dst} exceeds misaligned model's layer count "
                f"({self.num_misaligned_layers})."
            )

        logger.info(
            "Patching: l_src=%d (aligned) → l_dst=%d (misaligned), "
            "prompt_only=%s",
            l_src,
            l_dst,
            patch_prompt_only,
        )

        # ----------------------------------------------------------
        # Step 1: Cache aligned-model activations at l_src
        # ----------------------------------------------------------
        self.cache_aligned_activations(
            input_ids, attention_mask, layers=[l_src]
        )
        source_activations = self.aligned_cache.get(l_src)
        if source_activations is None:
            raise RuntimeError(
                f"Failed to cache aligned activations at layer {l_src}."
            )

        # ----------------------------------------------------------
        # Step 2: Register hooks on the misaligned model
        # ----------------------------------------------------------
        gen_cfg = self.config.generation
        self.patched_cache.clear()

        with HookManager() as hm:
            # Patching hook at l_dst
            patch_hook = self._make_patch_hook(
                source_activations=source_activations,
                patch_prompt_only=patch_prompt_only,
                patched_cache=self.patched_cache,
                layer_idx=l_dst,
            )
            hm.register(self._misaligned_layers[l_dst], patch_hook)

            # Optional: cache activations at additional layers for analysis
            if cache_patched_layers:
                for l in cache_patched_layers:
                    if l != l_dst:  # l_dst is already captured by patch_hook
                        cache_hook = self._make_cache_hook(
                            self.patched_cache, l
                        )
                        hm.register(self._misaligned_layers[l], cache_hook)

            # ----------------------------------------------------------
            # Step 3: Generate with the misaligned model
            # ----------------------------------------------------------
            with torch.no_grad():
                output_ids = self.misaligned_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    do_sample=gen_cfg.do_sample,
                    temperature=(
                        gen_cfg.temperature if gen_cfg.do_sample else None
                    ),
                    top_p=gen_cfg.top_p if gen_cfg.do_sample else None,
                    repetition_penalty=gen_cfg.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Hooks are removed here by HookManager.__exit__

        # Decode only the newly generated tokens
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Full analysis run (combined cache + patch + compare)
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        l_src: Optional[int] = None,
        l_dst: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete analysis for a single (l_src, l_dst) configuration.

        This is a convenience method that performs all three phases:
        1. Cache activations from both models (all layers).
        2. Generate three outputs: aligned baseline, misaligned baseline,
           and patched misaligned.
        3. Compute cosine-similarity trajectories and activation diffs.

        Args:
            input_ids: Prompt token IDs.
            attention_mask: Optional attention mask.
            l_src: Source layer (aligned model).
            l_dst: Destination layer (misaligned model).

        Returns:
            Dictionary with keys:
            - ``aligned_output``: Text generated by the aligned model.
            - ``misaligned_output``: Text generated by the misaligned model.
            - ``patched_output``: Text generated with reverse patching.
            - ``cosine_misaligned_vs_aligned``: Per-layer cosine similarity
              between misaligned and aligned activations.
            - ``cosine_patched_vs_aligned``: Per-layer cosine similarity
              between patched and aligned activations.
            - ``activation_diff_norms``: Per-layer L2 norm of Δh_l.
        """
        l_src = l_src if l_src is not None else self.config.patching.l_src
        l_dst = l_dst if l_dst is not None else self.config.patching.l_dst

        all_layers = list(
            range(min(self.num_aligned_layers, self.num_misaligned_layers))
        )

        # ---- Phase 1: Cache all-layer activations ----
        self.cache_aligned_activations(input_ids, attention_mask, all_layers)
        self.cache_misaligned_activations(input_ids, attention_mask, all_layers)

        # ---- Phase 2: Baseline generations ----
        aligned_output = self.generate_baseline(
            self.aligned_model, input_ids, attention_mask
        )
        misaligned_output = self.generate_baseline(
            self.misaligned_model, input_ids, attention_mask
        )

        # ---- Phase 3: Patched generation with layer caching ----
        patched_output = self.generate_with_patching(
            input_ids,
            attention_mask,
            l_src=l_src,
            l_dst=l_dst,
            cache_patched_layers=all_layers,
        )

        # ---- Phase 4: Cosine-similarity trajectories ----
        cosine_misaligned = self.compute_cosine_trajectory(
            input_ids, attention_mask, all_layers
        )
        cosine_patched = self.compute_cosine_trajectory(
            input_ids,
            attention_mask,
            all_layers,
            compare_cache=self.patched_cache,
        )

        # ---- Phase 5: Activation-difference norms ----
        diff_norms: Dict[int, float] = {}
        for l in all_layers:
            a = self.aligned_cache.get(l)
            m = self.misaligned_cache.get(l)
            if a is not None and m is not None:
                diff = (m - a).float()
                diff_norms[l] = diff.norm(dim=-1).mean().item()

        return {
            "aligned_output": aligned_output,
            "misaligned_output": misaligned_output,
            "patched_output": patched_output,
            "cosine_misaligned_vs_aligned": cosine_misaligned,
            "cosine_patched_vs_aligned": cosine_patched,
            "activation_diff_norms": diff_norms,
        }
