#!/usr/bin/env python

import argparse
import logging
import os

from typing import List
from subprocess import call

from framenet_tools.config import ConfigManager
from framenet_tools.evaluator import evaluate_frame_identification
from framenet_tools.pipeline import Pipeline
from framenet_tools.utils.static_utils import download, get_spacy_en_model

dirs = ["/scripts", "/lib", "/resources", "/data"]

required_files = [
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/scripts.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/resources.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z",
]


def check_files(path):
    logging.info(f"SRLPackage: Checking for required files:")

    for dir, required_file in zip(dirs, required_files):
        complete_path = path + dir

        if os.path.isdir(complete_path):
            logging.info(f"[Skip] Already found {complete_path}!")
        else:
            download(required_file)


def create_argparser():
    """
    Creates the ArgumentParser and defines all of its arguments.

    :return: the set up ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Provides tools to self-train and identify frames on raw text."
    )

    parser.add_argument(
        "action",
        help=f"Actions to perform, namely: download, convert, train, predict, evaluate",
    )
    parser.add_argument(
        "--level",
        help="The upper bound for pipeline stages. (Default is 4, meaning all stages!)",
        type=int,
    )
    parser.add_argument(
        "--path", help="A path specification used by some actions.", type=str
    )
    parser.add_argument(
        "--out_path", help="The path used for saving predictions", type=str
    )
    parser.add_argument(
        "--use_eval_files",
        help="Specify if eval files should be used for training as well.",
        action="store_true",
    )

    return parser


def eval_args(
    parser: argparse.ArgumentParser, cM: ConfigManager, args: List[str] = None
):
    """
    Evaluates the given arguments and runs to program accordingly.

    :param parser: The ArgumentParser for getting the specified arguments
    :param cM: The ConfigManager for getting necessary variables
    :return:
    """

    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    level = 4

    if parsed.level is not None:
        level = parsed.level

    if parsed.action == "download":

        if parsed.path is not None:
            check_files(os.path.join(os.getcwd(), parsed.download))
        else:
            check_files(os.getcwd())

        get_spacy_en_model()

    if parsed.action == "convert":

        call(
            [
                "pyfn",
                "convert",
                "--from",
                "fnxml",
                "--to",
                "semafor",
                "--source",
                "data/fndata-1.5-with-dev",
                "--target",
                "data/experiments/xp_001/data",
                "--splits",
                "train",
                "--output_sentences",
            ]
        )

        for dataset in ["train", "dev", "test"]:
            call(
                [
                    "pyfn",
                    "convert",
                    "--from",
                    "fnxml",
                    "--to",
                    "semeval",
                    "--source",
                    "data/fndata-1.5-with-dev",
                    "--target",
                    "data/experiments/xp_001/data",
                    "--splits",
                    dataset,
                ]
            )

    if parsed.action == "train":

        pipeline = Pipeline(cM, level)

        if parsed.use_eval_files:
            pipeline.train(cM.all_files)
        else:
            pipeline.train(cM.train_files)

    if parsed.action == "predict":

        if parsed.path is None:
            raise Exception("No input file for prediction given!")

        if parsed.out_path is None:
            raise Exception("No path specified for saving!")

        pipeline = Pipeline(cM, level)

        pipeline.predict(parsed.path, parsed.out_path)

    if parsed.action == "evaluate":

        pipeline = Pipeline(cM, level)

        pipeline.evaluate()


def main():
    """
    The main entry point

    :return:
    """

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

    cM = ConfigManager()
    parser = create_argparser()

    eval_args(parser, cM)


logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

cM = ConfigManager()
#pipeline = Pipeline(cM, 2)
print(cM.semeval_files)
#pipeline.train(cM.train_files)
evaluate_frame_identification(cM)