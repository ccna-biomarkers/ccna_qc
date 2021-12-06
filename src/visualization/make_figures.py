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
import features.build_features
import utils.utils as utils


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Documentation at https://github.com/ccna-biomarkers/ccna_qc_summary
    """)

    parser.add_argument(
        "-i", "--input_dir", required=False, default=".", help="Input BIDS directory, required if `--input-file` not defined.",
    )

    parser.add_argument(
        "-f", "--input-file", required=False, help="Path to the generated QC summary metrics (.csv), by default metrics are computed and saved at \"./data/processed/qc_metrics.csv\".",
    )

    parser.add_argument(
        "-o", "--output-dir", required=False, help="Output figure data directory (default: \"./reports/figures\")",
    )

    parser.add_argument(
        "--version", action="version", version=utils.get_version()
    )

    return parser


def violin_plot(metrics, group="task", metric_to_plot="fds_mean", figure_title="CCNA QC analytics", output_dir="./reports/figures", figure_filename="fd_mean.png"):
    # sns.set(style="darkgrid")
    # sns.set(style="whitegrid")
    # sns.set_style("white")
    sns.set(style="whitegrid", font_scale=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax = pt.RainCloud(x=group, y=metric_to_plot, data=metrics,
                      palette="Set2", bw=0.2, width_viol=.5, ax=ax, orient="v")
    plt.title(figure_title)
    plt.show()

    output_figure_path = os.path.join(output_dir, figure_filename)
    fig.savefig(output_figure_path, bbox_inches="tight")


if __name__ == '__main__':
    output_filepath = os.path.join(
        ROOT_DIR, "..", "data/processed/qc_metrics.csv")
    args = get_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT_DIR, "..", "reports/figures")
    print(args)

    if args.input_file is not None:
        if os.path.exists(args.input_file):
            print("\tReading metrics from {}".format(args.input_file))
            metrics = pd.read_pickle(args.input_file)
        else:
            raise ValueError("{} does not exist!".format(args.input_file))
    else:
        print("\tExtracting BIDS metadata from {}".format(args.input_dir))
        metadata = features.build_features.get_metadata(args.input_dir)
        print("\tComputing metrics...")
        metrics = features.build_features.compute_qc_metrics(metadata)
        print("\tAutomatic QC and saving to {}".format(output_filepath))
        metrics = features.build_features.auto_quality_control(metadata)
        metrics.to_csv(output_filepath)
    print("\tPlotting metrics...")
    violin_plot(metrics, group="task", metric_to_plot="fds_mean_raw", figure_title="CCNA QC analytics - Framewise displacement",
                output_dir=args.output_dir, figure_filename="raw_fds_mean.png")
    violin_plot(metrics, group="task", metric_to_plot="fds_mean_scrubbed", figure_title="CCNA QC analytics - Framewise displacement",
                output_dir=args.output_dir, figure_filename="scrubbed_fds_mean.png")
    violin_plot(metrics, group="datatype", metric_to_plot="dice", figure_title="CCNA QC analytics - Dice coefficient",
                output_dir=args.output_dir, figure_filename="dice.png")
