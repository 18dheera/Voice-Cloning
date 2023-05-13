# This script is used to preprocess audio files from datasets, encode them as mel spectrograms, and write them to disk. Audio files are also saved, to be used by the vocoder for training.

# It imports the necessary modules such as preprocess_dataset from synthesizer.preprocess, hparams from synthesizer.hparams, and print_args from utils.argutils. It also imports Path and argparse modules for parsing command-line arguments.

# In the if __name__ == "__main__": block, it creates an argument parser with several arguments such as datasets_root, out_dir, n_processes, skip_existing, hparams, no_alignments, datasets_name, and subfolders.

# It sets the default value of out_dir to <datasets_root>/SV2TTS/synthesizer/ and creates directories if they don't exist. It then calls the preprocess_dataset function with the parsed arguments. Finally, it prints the parsed arguments using the print_args function.


from synthesizer.preprocess import preprocess_dataset
from synthesizer.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_alignments", action="store_true", help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("--datasets_name", type=str, default="LibriSpeech", help=\
        "Name of the dataset directory to process.")
    parser.add_argument("--subfolders", type=str, default="train-clean-100,train-clean-360", help=\
        "Comma-separated list of subfolders to process inside your dataset directory")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)
    preprocess_dataset(**vars(args))
