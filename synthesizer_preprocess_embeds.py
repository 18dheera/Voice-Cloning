# This script creates embeddings for the synthesizer from the audio files in the LibriSpeech dataset. It takes three arguments:

# synthesizer_root: Path to the directory containing the audio files and train.txt file. By default, it should be <datasets_root>/SV2TTS/synthesizer/.
# encoder_model_fpath: Path to the trained encoder model.
# n_processes: Number of parallel processes to use. An encoder is created for each process, so if your GPU has low memory, you may need to lower this value. If you encounter CUDA errors, set it to 1.
# The script uses the create_embeddings function from the synthesizer.preprocess module to create the embeddings.


from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()

    # Preprocess the dataset
    print_args(args, parser)
    create_embeddings(**vars(args))
