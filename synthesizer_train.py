# This is a script that trains the neural network for the synthesizer using the ground truth mel spectrograms, the wavs and the embeddings. It takes several arguments through the command line, such as the name for the model, the path to the directory with the synthesizer data, the path to the output directory that will contain the saved model weights and the logs, and hyperparameter overrides.

# The train function is then called with these arguments. The function runs the training loop, which trains the neural network on the given data and saves the model weights and logs according to the specified parameters. If a model state from the same run ID was previously saved, the training will restart from there. The -f flag can be passed to overwrite saved states and restart from scratch.

from pathlib import Path
from synthesizer.hparams import hparams
from synthesizer.train import train
from utils.argutils import print_args
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("syn_dir", type=Path, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("-s", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    parser.add_argument("--hparams", default="", help=\
        "Hyperparameter overrides as a comma-separated list of name=value pairs")
    args = parser.parse_args()
    print_args(args, parser)

    args.hparams = hparams.parse(args.hparams)

    # Run the training
    train(**vars(args))
