#This is a Python script that runs a machine learning toolbox using the argparse module to handle command-line arguments.The script first imports necessary modules including the argparse module, os module, and the pathlib module.Then, the script defines the argparse object with several arguments including the datasets root, models directory, whether to use CPU for inference, and an optional seed value to make the toolbox deterministic.The script then parses the arguments and converts them to a dictionary. It prints the arguments and uses the 'cpu' flag to hide GPUs from PyTorch to force CPU processing.Finally, the script reminds the user to download pretrained models if necessary and launches the toolbox with the specified arguments using the Toolbox class.


import argparse
import os
from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
from utils.default_models import ensure_default_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.", default=None)
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models",
                        help="Directory containing all saved models")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, all inference will be done on CPU")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Remind the user to download pretrained models if needed
    ensure_default_models(args.models_dir)

    # Launch the toolbox
    Toolbox(**arg_dict)
