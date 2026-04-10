INTERVENTION_REGISTRY = {}

def register_intervention(name):
    """
    Register an intervention class under a unique name.

    This decorator adds the decorated class to ``INTERVENTION_REGISTRY``
    so it can later be instantiated through ``get_intervention``.

    Example:
        ```python
        @register_intervention("my_intervention")
        class MyIntervention(nn.Module):
            def __init__(self, base_model, tokenizer, scale=1.0):
                super().__init__()
                self.base_model = base_model
                self.tokenizer = tokenizer
                self.scale = scale

            def forward(self, input_ids, attention_mask):
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            def training_step(self, batch):
                return some_loss
        ```
    """
    def register_intervention_cls(cls):
        if name in INTERVENTION_REGISTRY:
            raise ValueError(f"Cannot register duplicate intervention '{name}'")
        INTERVENTION_REGISTRY[name] = cls
        return cls
    return register_intervention_cls

def get_intervention(name, *args, **kwargs):
    """
    Factory function to instantiate an intervention by name. \\
    The intervention name and parameters are defined in the training config.
    """
    if name not in INTERVENTION_REGISTRY:
        raise ValueError(
            f"Intervention '{name}' not found. "
            f"Available interventions: {list(INTERVENTION_REGISTRY.keys())}"
        )
    return INTERVENTION_REGISTRY[name](*args, **kwargs)
