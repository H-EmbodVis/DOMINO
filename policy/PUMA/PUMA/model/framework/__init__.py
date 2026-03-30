"""
Framework factory utilities.
Builds the PUMA framework from configuration.
"""

from PUMA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def build_framework(cfg):
    """
    Build the PUMA framework model from config.

    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Must be "PUMA"
    Returns:
        nn.Module: Instantiated PUMA model.
    """
    from PUMA.model.framework.PUMA import PUMA
    return PUMA(cfg)


__all__ = ["build_framework"]
