import os

# from copy import deepcopy # F401: deepcopy was imported but not used
from typing import (  # Ensured multi-line for robustness
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

from datasets import Dataset, load_dataset
from loguru import logger

# Assuming DataTrainingArguments has attributes like:
# dataset_name, dataset_config_name, data_files, train_file,
# validation_file, test_file
if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


LOGGER = logger
LABELS_MASK_VALUE = -100

__all__ = [
    "get_raw_dataset",
    "get_custom_datasets_from_path",
    "transform_dataset_keys",
]


def get_raw_dataset(
    dataset_args: "DataArgs",
    cache_dir: Optional[str] = None,
    streaming: Optional[bool] = False,
    **kwargs,
) -> Dataset:
    """
    Load the raw dataset from Hugging Face, using a cached copy if available.

    The specific dataset name, configuration, and data files are expected
    to be attributes of the `dataset_args` object.

    :param dataset_args: Arguments object containing dataset specifications
        (e.g., name, config, data files).
    :param cache_dir: Disk location to search for or save the cached dataset.
    :param streaming: True to stream data from Hugging Face; otherwise,
        download it.
    :return: The requested dataset(s).
    """
    # Construct arguments for load_dataset from dataset_args
    load_kwargs = {
        "path": dataset_args.dataset_name,
        "name": (
            dataset_args.dataset_config_name
            if hasattr(dataset_args, "dataset_config_name")
            else None
        ),
        "cache_dir": cache_dir,
        "streaming": streaming,
    }

    # Handle different ways data files can be specified
    if hasattr(dataset_args, "data_files") and dataset_args.data_files:
        load_kwargs["data_files"] = dataset_args.data_files
    elif (
        (hasattr(dataset_args, "train_file") and dataset_args.train_file)
        or (hasattr(dataset_args, "validation_file") and dataset_args.validation_file)
        or (hasattr(dataset_args, "test_file") and dataset_args.test_file)
    ):
        data_files = {}
        if hasattr(dataset_args, "train_file") and dataset_args.train_file:
            data_files["train"] = dataset_args.train_file
        if hasattr(dataset_args, "validation_file") and dataset_args.validation_file:
            data_files["validation"] = dataset_args.validation_file
        if hasattr(dataset_args, "test_file") and dataset_args.test_file:
            data_files["test"] = dataset_args.test_file
        load_kwargs["data_files"] = data_files

    load_kwargs.update(kwargs)

    # Filter out None values
    load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

    raw_datasets = load_dataset(**load_kwargs)
    return raw_datasets


def get_custom_datasets_from_path(
    path: str, ext: str = "json"
) -> Dict[str, Any]:  # Allow list of strings for values
    """
    Get a dictionary of custom datasets from a directory path.

    Supports Hugging Face's `load_dataset` for local folder datasets:
    https://huggingface.co/docs/datasets/loading

    This function scans the specified directory for files with a given
    extension (default '.json'). It builds a dictionary where keys are
    subdirectory names or dataset names, and values are file paths or
    lists of file paths.

    :param path: Path to the directory containing dataset files.
    :param ext: File extension to filter by (default 'json').
    :return: Dictionary mapping dataset names to file paths or lists of paths.
    Example:
        dataset_map = get_custom_datasets_from_path("./my_data", "jsonl")

    Note:
        If datasets are in subdirectories, keys are subdir names and values
        are lists of file paths. If directly in the main directory, keys are
        dataset names and values are single file paths.

    Accepts structures like:
        - path/
            train.json
            test.json
            val.json

        - path/
            train/
                data1.json
                data2.json
                ...
            test/
                ...
            val/
                ...
    """
    data_files: Dict[str, Any] = {}

    if not ext.startswith("."):
        ext = "." + ext

    if any(filename.endswith(ext) for filename in os.listdir(path)):
        for filename in os.listdir(path):
            if filename.endswith(ext):
                name, _ = os.path.splitext(filename)
                data_files[name] = os.path.join(path, filename)
    else:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                dir_name = item
                dir_dataset_files = []
                for filename in os.listdir(item_path):
                    if filename.endswith(ext):
                        file_path = os.path.join(item_path, filename)
                        dir_dataset_files.append(file_path)
                if dir_dataset_files:
                    data_files[dir_name] = (
                        dir_dataset_files[0]
                        if len(dir_dataset_files) == 1
                        else dir_dataset_files
                    )
    return transform_dataset_keys(data_files)


def transform_dataset_keys(data_files: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform dict keys to `train`, `val`, or `test` if a unique match exists.

    Example: `data_files = {"train_foo": "path/train.json"}` becomes
             `{"train": "path/train.json"}`.
    If multiple files could match (e.g., "train_foo", "train_bar"), no
    transformation for "train" occurs.

    :param data_files: The dict where keys might be transformed.
    :return: The dictionary with transformed keys.
    """
    keys_to_check = set(data_files.keys())

    dataset_split_names = ("train", "validation", "test")

    for target_split_name in dataset_split_names:
        matching_keys = [k for k in keys_to_check if target_split_name in k]

        if len(matching_keys) == 1:
            original_key = matching_keys[0]
            if original_key != target_split_name:
                data_files[target_split_name] = data_files.pop(original_key)

    # Handle 'val' as an alias for 'validation' if 'validation' is not
    # already present and a unique 'val' key exists.
    # Check 'val' is not already a standard name from dataset_split_names
    if "validation" not in data_files and "val" not in dataset_split_names:
        # Ensure we don't re-process "validation" if it contains "val"
        val_matching_keys = [
            k for k in keys_to_check if "val" in k and k != "validation"
        ]

        # Refined check for 'val' keys that are not 'validation'.
        # This assumes 'val' keys are distinct enough not to be confused
        # with 'validation' if both exist.
        # A key is a "true val key" if it's exactly 'val' or starts
        # with 'val_' but does not start with 'validation'.
        true_val_keys = [
            k
            for k in val_matching_keys
            if k == "val" or (k.startswith("val_") and not k.startswith("validation"))
        ]

        # If there's a unique, distinct 'val' key
        if len(true_val_keys) == 1:
            original_val_key = true_val_keys[0]
            # If the key is 'val' itself, or some other form, and
            # 'validation' is not a key yet
            if "validation" not in data_files:
                data_files["validation"] = data_files.pop(original_val_key)

    return data_files


# Ensure there's a newline character at the end of this file
