from .tutte_template import TutteTemplate
from .tutte_model import TutteModel


__all__ = {
    "TutteModel": TutteModel,
    "TutteTemplate": TutteTemplate,
}

def build_tutte_model(model_cfg, runtime_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset
    )

    return model
