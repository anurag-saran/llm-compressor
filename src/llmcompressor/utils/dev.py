import contextlib
import logging  # For interacting with transformers library's logger
import os
import tempfile
from typing import Type

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

from llmcompressor.utils.helpers import patch_attr

# from loguru import logger # Per F401, unused in this file's current state.


__all__ = ["skip_weights_download", "patch_transformers_logger_level"]


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager: initialize models without downloading model weight files.

    Differs from `init_empty_weights`: weights are allocated on assigned
    devices with random values, not on the meta device.

    :param model_class: class to patch, defaults to `AutoModelForCausalLM`
    """
    original_fn = model_class.from_pretrained
    # Files to ignore when downloading the model snapshot.
    weights_files_to_ignore = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        # e.g., "model.safetensors.index.json" from transformers.utils
        SAFE_WEIGHTS_INDEX_NAME,
        # e.g., "pytorch_model.bin.index.json" from transformers.utils
        WEIGHTS_INDEX_NAME,
        "*.msgpack",
    ]

    @classmethod
    def patched(cls, *args, **kwargs):
        nonlocal tmp_dir

        model_stub_arg = "pretrained_model_name_or_path"
        # Intercept model stub (pretrained_model_name_or_path)
        model_stub = args[0] if args else kwargs.pop(model_stub_arg)

        # Download files into tmp dir, ignoring specified weight patterns
        os.makedirs(tmp_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_stub,
            local_dir=tmp_dir,
            ignore_patterns=weights_files_to_ignore,
        )

        # Make an empty weights file to avoid errors during model loading.
        weights_file_path = os.path.join(tmp_dir, "model.safetensors")
        save_file({}, weights_file_path, metadata={"format": "pt"})

        # Load model from the temporary directory
        model = original_fn(tmp_dir, **kwargs)

        # Replace model_path attributes to reflect original stub
        model.name_or_path = model_stub
        if hasattr(model, "config") and model.config is not None:
            model.config._name_or_path = model_stub

        return model

    # Using explicit line continuation for the with statement
    with tempfile.TemporaryDirectory() as tmp_dir, patch_attr(
        model_class, "from_pretrained", patched
    ), skip_weights_initialize(), patch_transformers_logger_level():
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """
    Similar to `transformers.model_utils.no_init_weights`.

    Also patches torch.Tensor initialization to account for tensors
    not initialized on the meta device.
    """

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor  # Return tensor as is (random values from allocation)

    with contextlib.ExitStack() as stack:
        for name in TORCH_INIT_FUNCTIONS.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    """
    Context manager: modify Hugging Face transformers library logger level.

    Use with `skip_weights_download` to squelch transformers warnings
    about missing parameters in a checkpoint.

    :param level: new logging level for 'transformers' logger
                  (e.g., logging.WARNING, logging.ERROR).
    """
    # Get the specific logger used by Hugging Face transformers
    hf_transformers_logger = logging.getLogger("transformers")
    original_level = hf_transformers_logger.getEffectiveLevel()

    hf_transformers_logger.setLevel(level)
    try:
        yield
    finally:
        # Restore original logging level for the transformers logger
        hf_transformers_logger.setLevel(original_level)


# MAKE SURE THERE IS A NEWLINE CHARACTER AFTER THIS LINE
