import phate


def load_tree_data():
    tree_data, tree_clusters = phate.tree.gen_dla()
    return tree_data, tree_clusters
