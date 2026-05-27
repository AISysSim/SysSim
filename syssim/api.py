"""Public low-level API for SysSim."""

def set_efficiency_model_dir(model_dir: str) -> None:
    """Configure directory containing trained efficiency models.

    Args:
        model_dir: Path to directory with model files (*.pth).
    """
    from .compute.efficiency_models import set_backend_dir
    set_backend_dir(model_dir)
