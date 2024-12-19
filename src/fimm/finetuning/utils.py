import torch.nn as nn


def reset_head(model, args):
    """
    Reset the head of a model to the given number of classes.

    Args:
        model (nn.Module): The model to reset the head of.
        args: The arguments containing the number of classes to reset the head to.

    Returns:
        The model with the head reset to the given number of classes.
    """
    pool = None
    if hasattr(model, "global_pool"):
        pool = getattr(model, "global_pool")
        if not isinstance(pool, str):
            pool = pool.pool_type
    model.reset_classifier(num_classes=args.num_classes, global_pool=pool)


def prepare_model(model: nn.Module, args):
    """
    Create a fine-tuning model from a given model.

    Args:
        model (nn.Module): The model to create a fine-tuning model from.
        args: Arguments-object for creating the fine-tuning model.

    Returns:
        nn.Module: The created fine-tuning model.
    """
    # reset the classifier/head of the model
    reset_head(model, args)

    total_params: int = sum(param.numel() for param in model.parameters())

    # freeze model parameters
    if args.linear_probing:
        for param in model.parameters():
            param.requires_grad = False
    else:
        num_frozen: int = 0
        for param in model.parameters():
            if num_frozen < args.pct_to_freeze * total_params:
                param.requires_grad = False
                num_frozen += param.numel()
            else:
                break
    # unfreezing classifier/head, always used for finetuning
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total params: {total_params} | Trainable params: {trainable_params} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )
