# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
DSPy Compiler Configuration
============================
Offline re-optimization of pipeline modules using human-labeled feedback (Loop 3).
"""

import dspy
from app.core.config import settings
from app.core.debuglog import log

_dspy_configured = False


def configure_dspy():
    """Initialize DSPy with LiteLLM as the default language model."""
    global _dspy_configured

    model = settings.LITELLM_MODEL

    # Route the correct API key based on model prefix
    if "deepseek" in model.lower():
        api_key = settings.DEEPSEEK_API_KEY
    elif "claude" in model.lower() or "anthropic" in model.lower():
        api_key = settings.ANTHROPIC_API_KEY
    else:
        api_key = settings.OPENAI_API_KEY

    lm = dspy.LM(
        model=model,
        api_key=api_key,
    )

    if not _dspy_configured:
        log(f"DSPY: first configure(), model={model}")
        dspy.configure(lm=lm)
        _dspy_configured = True
    else:
        log(f"DSPY: already configured, using dspy.context()")
        dspy.context(lm=lm)

    return lm


def compile_module(module: dspy.Module, trainset: list[dspy.Example]) -> dspy.Module:
    """Run DSPy compiler (BootstrapFewShot or MIPROv2) on a module.

    Loop 3 [C4][J1]: User feedback → labeled examples → re-optimization.
    """
    optimizer_name = settings.DSPY_OPTIMIZER

    if optimizer_name == "BootstrapFewShot":
        optimizer = dspy.BootstrapFewShot(
            metric=None,  # Use default metric or pass custom
            max_labeled_demos=settings.DSPY_MAX_LABELED_EXAMPLES,
        )
    elif optimizer_name == "MIPROv2":
        optimizer = dspy.MIPROv2(
            metric=None,
            num_threads=4,
        )
    else:
        raise ValueError(f"Unknown DSPy optimizer: {optimizer_name}")

    compiled = optimizer.compile(module, trainset=trainset)
    return compiled
