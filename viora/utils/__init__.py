"""Cross-cutting utilities: config, logging, seeding, device management."""

from viora.utils.config import (
    VioraConfig,
    config_to_dict,
    default_config,
    load_config,
    save_config,
)
from viora.utils.device import (
    DeviceInfo,
    autocast_context,
    get_device_info,
    resolve_amp_dtype,
    resolve_device,
)
from viora.utils.logging import configure_logging, get_logger, log_dict
from viora.utils.seed import get_rng_states, seed_worker, set_rng_states, set_seed

__all__ = [
    "VioraConfig",
    "load_config",
    "save_config",
    "default_config",
    "config_to_dict",
    "resolve_device",
    "get_device_info",
    "resolve_amp_dtype",
    "autocast_context",
    "DeviceInfo",
    "configure_logging",
    "get_logger",
    "log_dict",
    "set_seed",
    "seed_worker",
    "get_rng_states",
    "set_rng_states",
]
