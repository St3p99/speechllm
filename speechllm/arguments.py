from dataclasses import dataclass, field
from typing import Dict, List, Optional

import transformers


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data config."}
    )
    subset: str = field(default=None, metadata={"help": "Subset to use for training."})
    split: str = field(default=None, metadata={"help": "Split to use for training."})
    amount: str = field(
        default=None, metadata={"help": "Amount of data to use for training."}
    )
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    remap_keys: Optional[Dict[str, str]] = None
    sampling_rate: Optional[int] = None
    num_proc_for_preprocessing: int = field(default=1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    freeze_modules: List[str] = field(default_factory=list)
