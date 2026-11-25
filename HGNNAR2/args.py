
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="gpu number or cpu")
    parser.add_argument("--path", type=str, default="./data/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="yelp", help="Dataset name")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=1, help="Repeated times")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    parser.add_argument("--ratio", type=float, default=0.2, help="Label ratio")
    parser.add_argument("--num_epoch", type=int, default=500, help="Training epochs")

    parser.add_argument("--alpha", type=float, default=1, help="Value of alpha")
    parser.add_argument("--beta", type=float, default=0.12, help="Value of beta")
    parser.add_argument("--tau", type=float, default=3.11, help="Value of tau")

    parser.add_argument("--hidden_dims", type=int, default=256, help="hidden dimensions")
    parser.add_argument("--emb_dims", type=int, default=256, help="hidden dimensions")

    parser.add_argument("--pretrain_ep", type=int, default=20)

    args = parser.parse_args()

    return args
