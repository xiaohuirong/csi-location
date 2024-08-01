import argparse
import datetime
from rich.table import Table
from rich.console import Console


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0, help="Round number.")
    parser.add_argument("--scene", type=int, default=1, help="Scene number.")

    # time stamp
    now = datetime.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
    parser.add_argument(
        "--time", type=str, default=now, help="the time when run this experiment."
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tseed", type=int, default=0, help="Torch random seed.")

    parser.add_argument(
        "--embedding", type=int, default=1024, help="The number of embedding layer."
    )

    parser.add_argument("--bsz", type=int, default=100, help="Batch size.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    parser.add_argument(
        "--gamma", type=float, default=0.9, help="Learing rate discount factor."
    )

    parser.add_argument(
        "--step", type=int, default=200, help="The step between two discount step."
    )

    parser.add_argument("--epoch", type=int, default=5000, help="Total epoch")

    parser.add_argument(
        "--test", action="store_true", default=False, help="Enable test or not."
    )
    parser.add_argument(
        "--slice",
        action="store_true",
        default=False,
        help="Whether use slice data or not.",
    )

    parser.add_argument("--cp", type=str, default="None", help="load parameter")

    args = parser.parse_args()

    r = args.round
    s = args.scene

    args.data_dir = f"data/round{r}/s{s}/data/"
    args.feature_dir = f"data/round{r}/s{s}/feature/"
    args.result_dir = f"data/round{r}/s{s}/result/"

    args.cfg_path = args.data_dir + f"Round{r}CfgData{s}.txt"
    args.txt_data_path = args.data_dir + f"Round{r}InputData{s}.txt"
    args.data_path = args.data_dir + f"Round{r}InputData{s}.npy"

    args.txt_pos_path = args.data_dir + f"Round{r}InputPos{s}.txt"

    args.pos_path = args.data_dir + f"Round{r}InputPos{s}_S.npy"
    args.index_path = args.data_dir + f"Round{r}Index{s}_S.npy"
    args.data_slice_path = args.data_dir + f"Round{r}InputData{s}_S.npy"

    if args.test:
        args.truth_pos_path = args.data_dir + f"Round{r}GroundTruth{s}.txt"

        args.test_data_slice_path = (
            args.data_dir + f"Test{args.seed}Round{r}InputData{s}_S.npy"
        )
        args.test_pos_path = args.data_dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"
        args.test_index_path = args.data_dir + f"Test{args.seed}Round{r}Index{s}_S.npy"

    args.feature_path = args.feature_dir + "F" + f"Round{r}InputData{s}.npy"
    args.feature_slice_path = args.feature_dir + "F" + f"Round{r}InputData{s}_S.npy"
    args.test_feature_slice_path = (
        args.feature_dir + "F" + f"Test{args.seed}Round{r}InputData{s}_S.npy"
    )

    args.pth_path = args.result_dir + f"M{args.tseed}Round{r}Scene{s}.pth"
    args.output_pos_path = args.result_dir + f"Round{r}OutputPos{s}.npy"
    args.txt_output_pos_path = args.result_dir + f"Round{r}OutputPos{s}.txt"

    return args


def show_args(args):
    # Convert args to a dictionary
    args_dict = vars(args)

    # Create a table
    table = Table(title="Command Line Arguments")

    # Add columns
    table.add_column("Argument", justify="center", style="cyan", no_wrap=True)
    table.add_column("Value", justify="center", style="magenta")

    # Add rows
    for key, value in args_dict.items():
        table.add_row(key, str(value))

    # Print the table
    console = Console()
    console.print(table)


if __name__ == "__main__":
    args = parse_args()
    show_args(args)
