"""RoPE scores measured during existing evaluation forwards."""
from .monitor import MonitorConfig, RoPEMonitor
from .scores import ScoreCache, ScoreResult, score_activations
from .sampling import SamplePlan, sample_tokens
from .loggers import JSONLLogger, WandbLogger
from .integrations.trainer import attach_rope_monitor
from ._version import __version__
from .regions import token_region_masks
from .heads import PaperHeadPreset, get_head_preset, list_head_presets

__all__ = ["MonitorConfig", "RoPEMonitor", "ScoreCache", "ScoreResult", "score_activations",
           "SamplePlan", "sample_tokens", "token_region_masks", "JSONLLogger", "WandbLogger", "attach_rope_monitor",
           "PaperHeadPreset", "get_head_preset", "list_head_presets"]
