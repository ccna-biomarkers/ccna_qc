#!/usr/bin/python3

import matplotlib.collections as clt
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns
import pandas as pd
import os
import sys
import argparse
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_DIR)
import utils.utils as utils
import features.build_features


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Documentation at https://github.com/ccna-biomarkers/ccna_qc_summary
    """)

    parser.add_argument(
        "-i", "--input_dir", required=False, default=".", help="Input BIDS directory, required if `--input-file` not defined.",
    )

    parser.add_argument(
        "-f", "--input-file", required=False, help="Path to the pickled (\".pkl\") generated QC summary metrics, if not defined metrics will be computed.",
    )

    parser.add_argument(
        "-o", "--output-dir", required=False, help="Output figure data directory (default: \"./reports/figures\")",
    )

    parser.add_argument(
        "--version", action="version", version=utils.get_version()
    )

    return parser


def violin_plot(confounds, output_dir="./reports/figures"):
    # sns.set(style="darkgrid")
    # sns.set(style="whitegrid")
    # sns.set_style("white")
    sns.set(style="whitegrid", font_scale=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax = pt.RainCloud(x="task", y="fds_mean", data=confounds,
                      palette="Set2", bw=0.2, width_viol=.5, ax=ax, orient="v")
    plt.title("CCNA QC analytics")
    plt.show()

    output_figure_path = os.path.join(output_dir, "fd_mean.png")
    fig.savefig(output_figure_path, bbox_inches="tight")


if __name__ == '__main__':
    output_filepath = os.path.join(ROOT_DIR, "..", "data/processed/confounds.pkl")
    args = get_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT_DIR, "..", "reports/figures")
    print(args)

    if args.input_file is not None:
        if os.path.exists(args.input_file):
            print("\tReading confounds from {}".format(args.input_file))
            confounds = pd.read_pickle(args.input_file)
        else:
            raise ValueError("{} does not exist!".format(args.input_file))
    else:
        print("\tExtracting confounds and saving to {}".format(output_filepath))
        confounds = features.build_features.get_confounds(args.input_dir)
        confounds.to_pickle(output_filepath)
    print("\tPlotting confounds...")
    violin_plot(confounds, output_dir=args.output_dir)
