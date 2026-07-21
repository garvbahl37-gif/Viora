"""Configuration system.

Viora is configuration-driven: every architectural dimension lives in a typed
dataclass, never hardcoded in a module. We use :mod:`dataclasses` for the schema
(so IDEs and ``mypy`` understand it) and :mod:`omegaconf` for YAML loading,
structured merging, and command-line dotlist overrides.

Usage::

    cfg = load_config("configs/model/viora_tiny.yaml",
                      overrides=["vision.depth=8", "llm.dummy=true"])
    save_config(cfg, "runs/exp1/resolved_config.yaml")

Only *model* configs live here. Training / data / evaluation configs are defined
next to their subsystems and composed in at the experiment level, so this file
stays the single source of truth for architecture without ballooning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Component configs                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class VisionConfig:
    """3D spatiotemporal Vision Transformer (tubelet embedding + ViT blocks)."""

    # backbone: "scratch" = Viora's 3D ViT (trained from scratch);
    #           "pretrained" = a frozen pretrained image encoder (SigLIP/CLIP), the pragmatic path.
    backbone: str = "scratch"
    pretrained_name: str = "google/siglip-base-patch16-224"
    pretrained_pool: str = "mean"
    freeze_backbone: bool = True

    image_size: int = 224
    num_frames: int = 16
    in_channels: int = 3
    tubelet_size: int = 2  # temporal patch depth
    patch_size: int = 16  # spatial patch (square)
    dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    attention_mode: str = "factorized"  # "factorized" | "full"
    norm_type: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp_type: str = "mlp"  # "mlp" | "swiglu"
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0  # stochastic depth (max, linearly scaled by depth)
    pos_embed_spatial: str = "learnable"  # "learnable" | "sincos" | "none"
    pos_embed_temporal: str = "learnable"  # "learnable" | "sincos" | "none"
    use_sdpa: bool = True  # torch.nn.functional.scaled_dot_product_attention
    gradient_checkpointing: bool = False


@dataclass
class TemporalConfig:
    """Hierarchical temporal encoder operating on per-clip temporal tokens."""

    dim: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    local_window: int = 4  # local temporal attention window (in tokens)
    global_layers: int = 2  # number of global (full temporal) layers among depth
    spatial_pool: str = "mean"  # collapse spatial tokens per frame: "mean" | "attention" | "cls"
    norm_type: str = "layernorm"
    mlp_type: str = "mlp"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_sdpa: bool = True
    gradient_checkpointing: bool = False


@dataclass
class EventConfig:
    """Event tokenizer: compress temporal features into semantic event tokens."""

    num_queries: int = 16
    dim: int = 384
    num_heads: int = 6
    num_layers: int = 2
    mlp_ratio: float = 4.0
    norm_type: str = "layernorm"
    use_sdpa: bool = True


@dataclass
class MemoryConfig:
    """Bounded, timestamp-aware temporal memory for long-video understanding."""

    enabled: bool = True
    dim: int = 384
    short_term_size: int = 16  # most-recent event tokens kept verbatim
    long_term_size: int = 32  # compressed long-term slots
    max_tokens: int = 48  # hard budget = short_term_size + long_term_size (validated)
    eviction: str = "importance"  # "fifo" | "importance"
    compression: str = "mean"  # how evicted events fold into long-term: "mean" | "drop"


@dataclass
class ResamplerConfig:
    """Perceiver/Q-Former-style resampler producing a fixed number of visual tokens."""

    num_queries: int = 32
    dim: int = 384  # operates in vision hidden dim
    num_heads: int = 6
    num_layers: int = 2
    mlp_ratio: float = 4.0
    norm_type: str = "layernorm"
    use_sdpa: bool = True


@dataclass
class ProjectorConfig:
    """Maps resampler output (vision dim) into the LLM embedding space."""

    type: str = "mlp"  # "linear" | "mlp"
    input_dim: int = 384  # = resampler.dim; resolved at build time
    output_dim: int = 0  # = llm hidden size; 0 means "resolve from LLM"
    hidden_dim: int = 0  # 0 -> defaults to output_dim (for mlp)
    depth: int = 2  # mlp layers
    activation: str = "gelu"


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class LLMConfig:
    """Language model adapter configuration."""

    name_or_path: str = ""  # HF causal LM id; empty + dummy=True for tests
    dummy: bool = False  # use the built-in tiny dummy decoder (tests / smoke)
    train_mode: str = "frozen"  # "frozen" | "lora" | "full"
    hidden_size: int = 0  # resolved from HF config; set explicitly for dummy
    vocab_size: int = 0  # resolved from tokenizer; set explicitly for dummy
    max_length: int = 512
    video_token: str = "<video>"  # placeholder consumed by visual-token injection
    dtype: str = "float32"  # "float32" | "bfloat16" | "float16"
    attn_implementation: str = "eager"  # "eager" | "sdpa" | "flash_attention_2"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class GroundingConfig:
    """Temporal grounding head (discretised temporal-bin formulation)."""

    enabled: bool = True
    num_bins: int = 64  # temporal bins spanning [0, duration]
    hidden_dim: int = 384


@dataclass
class VioraConfig:
    """Top-level model configuration composing all subsystems."""

    name: str = "viora_tiny"
    vision: VisionConfig = field(default_factory=VisionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    event: EventConfig = field(default_factory=EventConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    # Active task heads / objectives (drives which heads are built & losses used).
    tasks: list[str] = field(
        default_factory=lambda: ["video_qa", "captioning"]
    )


# --------------------------------------------------------------------------- #
# Load / save / merge machinery                                               #
# --------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    """Training / optimization configuration (loaded against experiment YAMLs)."""

    stage: int = 0
    output_dir: str = "runs/dev"
    seed: int = 0
    deterministic: bool = False

    # optimization
    max_steps: int = 1000
    epochs: int = 0              # 0 -> use max_steps
    batch_size: int = 2
    grad_accum: int = 1
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_steps: int = 100
    scheduler: str = "cosine"    # "cosine" | "constant" | "linear"
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # precision / memory
    precision: str = "fp32"      # "fp32" | "bf16" | "fp16"
    gradient_checkpointing: bool = False

    # freezing
    freeze_vision: bool = False
    freeze_llm: bool = True

    # data
    num_workers: int = 2
    datasets: list[str] = field(default_factory=list)
    mixture_weights: dict = field(default_factory=dict)
    mixture_temperature: float = 1.0
    loss_weights: dict = field(default_factory=dict)

    # logging / checkpointing
    log_every: int = 10
    eval_every: int = 0          # 0 disables periodic eval
    save_every: int = 500
    keep_last_checkpoints: int = 3  # prune older step_*.pt (0 = keep all); 'final.pt' is never pruned
    resume: str = ""             # checkpoint path to resume from
    tensorboard: bool = False

    # distributed (torchrun sets rank/world_size env; these pick the wrapper)
    ddp: bool = False
    fsdp: bool = False           # shard params across GPUs (for large LLMs); CUDA-only


def default_config() -> VioraConfig:
    """Return the default (viora_tiny-scale) config as a dataclass instance."""
    return OmegaConf.to_object(OmegaConf.structured(VioraConfig))  # type: ignore[return-value]


def load_config(
    path: str | Path | None = None,
    *,
    overrides: list[str] | None = None,
    schema: type = VioraConfig,
) -> Any:
    """Load a config: structured defaults <- YAML file <- CLI dotlist overrides.

    Args:
        path: YAML file to merge over the dataclass defaults. ``None`` uses defaults.
        overrides: OmegaConf dotlist strings, e.g. ``["vision.depth=8"]``.
        schema: The dataclass schema to structure against (default ``VioraConfig``).

    Returns:
        A dataclass instance of ``schema`` with all fields resolved and validated
        against the schema (unknown keys raise, wrong types raise).
    """
    base = OmegaConf.structured(schema)
    if path is not None:
        file_cfg = OmegaConf.load(str(path))
        base = OmegaConf.merge(base, file_cfg)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    # to_object validates types and returns concrete dataclass instances.
    return OmegaConf.to_object(base)


def save_config(cfg: Any, path: str | Path) -> None:
    """Serialize a resolved config to YAML (for experiment reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = OmegaConf.structured(cfg) if not OmegaConf.is_config(cfg) else cfg
    OmegaConf.save(conf, str(path))


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert any config (dataclass or OmegaConf) to a plain nested dict."""
    conf = OmegaConf.structured(cfg) if not OmegaConf.is_config(cfg) else cfg
    return OmegaConf.to_container(conf, resolve=True)  # type: ignore[return-value]
