import argparse
import datetime
from rich.table import Table
from rich.console import Console


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-pos", type=str, default="data/Round0GroundTruth1.txt",
        help="True positions.")
    parser.add_argument("--input-pos", type=str, default="data/Round0InputPos1.txt",
        help="Input positions.")
    parser.add_argument("--input-data", type=str, default="data/Round0InputData1.txt",
        help="Input channel datas.")
    parser.add_argument("--cfg-data", type=str, default="data/Round0CfgData1.txt",
        help="Config file.")

    # time stamp
    now = datetime.datetime.now().strftime("%y-%m-%d__%H-%M-%S")
    parser.add_argument("--time", type=str, default=now,
        help="the time when run this experiment.")

    parser.add_argument("--seed", type=int, default=0,
        help="Random seed.")
    # parser.add_argument("--tseed", type=int, default=0,
    #     help="torch seed")
    # parser.add_argument("--alpha", type=float, default=3,
    #     help="the learning rate of the optimizer")
    # parser.add_argument("--beta", type=float, default=1e-3,
    #     help="path loss in d_0")
    # parser.add_argument("--sigma", type=float, default=1e-6,
    #     help="noise power square root")

    # parser.add_argument("--lr", type=float, default=1e-4,
    #     help="the initial learning rate")

    # parser.add_argument("--rice_factor", type=float, default=5,
    #     help="rice factor")
    # parser.add_argument("--scaling_factor", type=float, default=1e6,
    #     help="scaling factor")
    # parser.add_argument("--value_factor", type=float, default=1,
    #     help="value scaling factor")
    # parser.add_argument("--threshold", type=float, default=1e-5,
    #     help="threshold")

    # parser.add_argument("--clu_num", type=int, default=5,
    #     help="number of cluster")
    # parser.add_argument("--node_num", type=int, default=5,
    #     help="number of node in each cluster")
    # parser.add_argument("--ant_num", type=int, default=8,
    #     help="number of antenna in each base station")

    # parser.add_argument("--clu_min_radius", type=float, default=1000,
    #     help="cluster min radius")
    # parser.add_argument("--clu_max_radius", type=float, default=2000,
    #     help="cluster max radius")
    # parser.add_argument("--node_min_radius", type=float, default=100,
    #     help="node min radius")
    # parser.add_argument("--node_max_radius", type=float, default=1000,
    #     help="node max radius")

    # parser.add_argument("--cp", type=str, default="none",
    #     help="saved checkpoint file.")
    # parser.add_argument("--cp2", type=str, default="none",
    #     help="saved checkpoint file.")
    # parser.add_argument('--isclip', action='store_true', default=False,
    #                     help='A boolean flag to indicate if clip')

    # parser.add_argument("--db", type=str, default="db/test.db",
    #     help="selected datebase")
    # parser.add_argument("--table", type=str, default="ite",
    #     help="selected datebase tabel name")

    # parser.add_argument("--method", type=str, default="sca",
    #     help="the solve method")
    # parser.add_argument("--topo", type=str, default="random",
    #     help="the method to generate topology")

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
