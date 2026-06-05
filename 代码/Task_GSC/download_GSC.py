import argparse
import os

from torchaudio import datasets


def main():
    parser = argparse.ArgumentParser(description="Download Google Speech Commands v0.02")
    parser.add_argument("--root", type=str, default=os.path.join(".", "data"), help="Root data directory")
    args = parser.parse_args()

    datasets.SPEECHCOMMANDS(
        root=args.root,
        url="speech_commands_v0.02",
        folder_in_archive="SpeechCommands",
        download=True,
    )
    print("Download finished:", os.path.abspath(args.root))


if __name__ == "__main__":
    main()
