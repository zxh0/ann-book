from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models
REF_PATH = ROOT.parent / "reference" / "smollm2_135m_forward.pt"


@pytest.fixture(scope="session")
def llama_ref():
    """HuggingFace's Llama modules, used as the numerical oracle.

    Skip on *any* failure, not just ImportError: transformers 5.x imports fine
    but disables its torch backend when torch < 2.5 (which is forced on macOS
    x86_64), and then blows up with NameError deep inside. A broken oracle
    should skip the comparison, not fail it.
    """
    try:
        from transformers.models.llama import modeling_llama

        assert modeling_llama.LlamaRMSNorm is not None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        pytest.skip(f"transformers oracle unavailable: {type(exc).__name__}: {exc}")
    return modeling_llama


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Path to the local SmolLM2-135M checkpoint.

    `models/` is gitignored, so a fresh clone has no weights — skip rather than
    fail so the rest of the suite still runs.
    """
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip(f"no checkpoint at {MODEL_DIR} — see CLAUDE.md for how to download it")
    return MODEL_DIR


@pytest.fixture(scope="module")
def reference():
    """Tensors dumped by `steps/00_reference.py`, used as the forward-pass oracle.

    `reference/` is gitignored and regenerable, so skip rather than fail.
    """
    if not REF_PATH.exists():
        pytest.skip("run steps/00_reference.py first to generate reference tensors")
    return torch.load(REF_PATH, weights_only=True)
