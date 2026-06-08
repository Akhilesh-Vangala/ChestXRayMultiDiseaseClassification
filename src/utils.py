import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any
import json
import logging
import random


def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def setup_logging(log_dir: str = 'outputs/logs'):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_config(config: Dict[str, Any], filepath: str):
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model: tf.keras.Model) -> int:
    return model.count_params()
