"""Optimizer groups for independent retrieval and geographic learning rates."""


def separate_learning_rate_groups(model, backbone_lr: float, geo_lr: float) -> list[dict]:
    """All unfrozen SigLIP parameters use backbone_lr; geo head/attention use geo_lr."""
    backbone = []
    geographic = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone.append(parameter)
        else:
            geographic.append(parameter)
    if not backbone and not geographic:
        raise ValueError("No trainable model parameters")
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if geographic:
        groups.append({"params": geographic, "lr": geo_lr})
    return groups
