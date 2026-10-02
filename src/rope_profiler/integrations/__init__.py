"""Training-framework adapters with optional third-party dependencies."""

from .trainer import TrainerMonitorAttachment, attach_rope_monitor

__all__ = ["TrainerMonitorAttachment", "attach_rope_monitor"]
