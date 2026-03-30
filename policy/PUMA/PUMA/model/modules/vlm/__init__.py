
def get_vlm_model(config):
    if "Qwen3-VL" in config.framework.qwenvl.base_vlm:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    else:
        raise ValueError(f"Unsupported VLM: {config.framework.qwenvl.base_vlm}")
