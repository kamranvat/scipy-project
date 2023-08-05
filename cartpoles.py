"""
Train different policies in the Cartpole v1 environment from stable baselines 3.
Outputs get logged as csv files. 
The results can be visualized and displayed.
"""
import argparse
from training import train_active_models, train_demo
from cli import toggle_active_models, load_settings
from comparison import compare

# Define a parser and command line arguments
parser = argparse.ArgumentParser(
    description="Train and compare different policies (with default hyperparameters) in the cartpole-v1 environment from stable-baselines3.",
    epilog="Warning: A high episode count over many models is computationally expensive.",
)

parser.add_argument(
    "--set",
    "-s",
    action="store_true",
    help="view/modify the list of models to be trained",
)

parser.add_argument(
    "--train",
    "-t",
    action="store_true",
    help="train all models set to 'active' in the model list",
)

parser.add_argument(
    "--compare",
    "-c",
    action="store_true",
    help="show only the visualized results from the last run, suppresses --set and --train",
)

parser.add_argument(
    "--demo",
    "-d",
    action="store_true",
    help="demo mode, trains A2C, then shows the trained agent in render mode 'human'. Does not log.",
)

parser.add_argument(
    "--runs",
    "-r",
    type=int,
    default=10000,
    help="amount of steps each agent should be trained for (default: 10,000)",
)

args = parser.parse_args()


if __name__ == "__main__":
    model_list = load_settings()

    if args.runs < 4000:
        print("Minimum value for timesteps is 4000.")
        args.runs = 4000

    if args.demo:
        train_demo(args.runs)
    else:
        if args.set:
            toggle_active_models(model_list)

        if args.train:
            train_active_models(model_list, args.runs)

        if args.compare:
            compare(model_list)
