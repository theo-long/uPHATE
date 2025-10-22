import argparse
import phate
import scprep
import matplotlib.pyplot as plt

from uphate.utils import load_tree_data


def main(args):
    if args.dataset == "tree":
        data, labels = load_tree_data()

    phate_operator = phate.PHATE(k=15, t=100, mds_solver="smacof", verbose=2)
    phate_output = phate_operator.fit_transform(data)
    scprep.plot.scatter2d(phate_output, c=labels)
    phate_operator.set_params(n_components=3)
    phate_output = phate_operator.transform()
    scprep.plot.rotate_scatter3d(phate_output, c=labels)

    plt.show(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tree")
    parser.add_argument("-t", default=None)
    parser.add_argument("--jax", action="store_true")
    args = parser.parse_args()
    main(args)
