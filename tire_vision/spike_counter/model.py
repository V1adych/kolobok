from torch import nn

from mmseg.apis import init_model


models = {
    "segformer": "tire_vision/spike_counter/configs/segformer_config.py",
    "bisenetv2": "tire_vision/spike_counter/configs/bisenetv2_config.py",
}


def get_model(model_name: str = "bisenetv2") -> nn.Module:
    return init_model(models[model_name], device="cpu")
