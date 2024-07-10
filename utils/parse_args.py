import argparse
import datetime
from rich.table import Table
from rich.console import Console


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0,
        help="Round number.")
    parser.add_argument("--scene", type=int, default=1,
        help="Scene number.")

    # time stamp
    now = datetime.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
    parser.add_argument("--time", type=str, default=now,
        help="the time when run this experiment.")

    parser.add_argument("--seed", type=int, default=0,
        help="Random seed.")
    parser.add_argument("--tseed", type=int, default=0,
        help="Torch random seed.")

    parser.add_argument("--embedding", type=int, default=1024,
        help="The number of embedding layer.")

    parser.add_argument("--bsz", type=int, default=100,
        help="Batch size.")

    parser.add_argument("--data", type=str, default="None",
        help="Data name.")

    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate.")

    parser.add_argument("--gamma", type=float, default=0.9,
        help="Learing rate discount factor.")

    parser.add_argument("--step", type=int, default=200,
        help="The step between two discount step.")

    parser.add_argument("--epoch", type=int, default=5000,
        help="Total epoch")

    parser.add_argument('--test', action='store_true', default=False, 
                        help='Enable test or not.')

    parser.add_argument("--port", type=int, default=0,
        help="Port index")

    parser.add_argument("--over", type=int, default=0,
        help="Overtension index")

    parser.add_argument("--method", type=int, default=1,
        help="Solve method")

    args = parser.parse_args()

    # fmt: on
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
