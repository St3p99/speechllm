

import subprocess
import sys
from typing import List, Union
from pathlib import Path
import torch
import os
import yaml


from speechllm.arguments import DataArguments
from speechllm.dataset.speech_dataset import SpeechDataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def launch_train(path_to_yaml_config: Union[str, Path]):
    with open(path_to_yaml_config, "r") as config_file:
        config_dict = yaml.safe_load(config_file)


    is_main_process = os.getenv("SLURM_PROCID", "0") == "0"
    if is_main_process:
        # load the dataset to pre-cache it
        logger.info("Loading dataset to pre-cache it")
        SpeechDataset(
            tokenizer=None, data_args=DataArguments(**config_dict["data"])
        )

    num_gpus = torch.cuda.device_count()
    num_nodes = 1

    train_command = [
        "accelerate", "launch",
            "--config-file", config_dict["accelerate_config"],
            "--deepspeed-config-file", config_dict["deepspeed_config"],
            "--num-processes", str(num_gpus),
            "--num-machines", str(num_nodes),
            "--machine-rank", os.getenv("SLURM_PROCID", "0"),
            "--main-process-ip", os.getenv("MASTER_ADDR", "localhost"),
            "--main-process-port", os.getenv("MASTER_PORT", "25000"),
            "speechllm/train.py",
            "--config", str(path_to_yaml_config),
    ]
    run_subprocess(train_command)



def run_subprocess(command: List[str]) -> None:
    # start the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # read process output in real-time
    while process.poll() is None:  # process hasn't finished yet
        output = process.stdout.readline()
        if output:
            print(output.strip("\n"), file=sys.stdout)

    # capture any remaining output after the process has finished
    for output in process.stdout:
        print(output.strip("\n"), file=sys.stdout)

    # return the exit code
    return process.returncode

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="/net/tscratch/people/plgstefanop/speechllm/speechllm/config/test1.yaml")
    args = parser.parse_args()

    launch_train(args.config)