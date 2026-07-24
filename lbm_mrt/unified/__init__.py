"""Unified model abstraction layer for lbm_mrt.

Provides a model-registry pattern that maps high-level physics descriptions
(collision model, EOS, force scheme, wetting) to the flat params.txt keys
expected by the CUDA solver. Built on top of the existing single_run.py
infrastructure — no CUDA code changes required.

Usage::

    from lbm_mrt.unified import ModelRegistry, run_scmp

    # List available SCMP models
    print(ModelRegistry.list_scmp())

    # Run a pre-registered model
    run_scmp("scmp_cs_huang_256", geom="data/geometry/droplet.plt", steps=50000)

    # Create a custom model
    from lbm_mrt.unified.components import EOSParams, ForceParams
    my_model = ModelDefinition(
        name="my_scmp",
        eos=EOSParams.carnahan_starling(a=1, b=4, R=1, T=0.066),
        force=ForceParams.huang_zhang(),
        ...
    )
"""

from .components import (
    CollisionParams,
    CollisionType,
    EOSParams,
    EOSType,
    EOS_DEFAULT_PARAMS,
    ForceParams,
    ForceType,
    WettingParams,
    WettingType,
)
from .models import (
    ModelDefinition,
    ModelRegistry,
    SCMP_MODELS,
)
from .runner import (
    run_model,
    run_scmp,
    validate_model_definition,
    print_validation_report,
)

__all__ = [
    # Components
    "CollisionParams",
    "CollisionType",
    "EOSParams",
    "EOSType",
    "EOS_DEFAULT_PARAMS",
    "ForceParams",
    "ForceType",
    "WettingParams",
    "WettingType",
    # Models
    "ModelDefinition",
    "ModelRegistry",
    "SCMP_MODELS",
    # Runner
    "run_model",
    "run_scmp",
    "validate_model_definition",
    "print_validation_report",
]
