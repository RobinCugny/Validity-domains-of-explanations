import argparse
import pickle
import numpy as np

from utils import grid_search_anchors_v3, load_dataset, train_model


def main(dataset = "flame", model_name = "m0", deltas = "0.05,0.1", taus = "0.1,0.15", Bs = "10,50,100", threshold_prec = "1", tries = "5"):
    X_train, X_test, y_train, y_test, X, y, feature_names = load_dataset(
        dataset)
    model = train_model(X_train, y_train, model_name)
    print(model.score(X, y))
    print(model.score(X_test, y_test))

    deltas  = list(map(float, str(deltas).split(",")))
    taus = list(map(float, str(taus).split(",")))
    Bs = list(map(int, str(Bs).split(",")))
    threshold_prec = int(threshold_prec)
    tries = np.linspace(1, int(tries), int(tries), dtype=int)

    results_grid_anchors = grid_search_anchors_v3(X, y, X_train, dataset, feature_names, model, deltas, taus, Bs, threshold_prec, tries)
    pickle.dump(results_grid_anchors, open("results_exp/"+dataset+"_anchors.p", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset', help='Name of the selected dataset', default="flame")
    parser.add_argument(
        '-m', '--model', help='Name of the selected model', default="m0")
    parser.add_argument('-d', '--deltas', help='Values to test for deltas hp', default="0.05,0.1")
    parser.add_argument('-t', '--taus', help='Values to test for taus hp', default="0.1,0.15")
    parser.add_argument('-b', '--Bs', help='Values to test for Bs hp', default="10,50,100")
    parser.add_argument('-p', '--prec', help='Values to test for threshold_prec hp', default="1")
    parser.add_argument('-s', '--tries', help='Number of tries for anchors seed', default="5")
    
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    deltas = args.deltas
    taus = args.taus
    Bs = args.Bs
    prec = args.prec
    tries = args.tries

    main(dataset, model, deltas, taus, Bs, prec, tries)

