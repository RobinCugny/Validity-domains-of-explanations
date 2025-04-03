import argparse
import pickle

import numpy as np

from utils import grid_search_VD_v3, load_dataset, train_model


def best_hp_VD(mean_precs, mean_cov, mean_dist_exp, empty_VDs):
    return np.argmax(np.asarray(mean_precs)+np.asarray(mean_cov)-np.asarray(mean_dist_exp)-np.asarray(empty_VDs))


def main(dataset = "flame", model_name = "m0", gammas = "20", tds = "1"):
    X_train, X_test, y_train, y_test, X, y, feature_names = load_dataset(
        dataset)
    print("Training model")
    model = train_model(X_train, y_train, model_name)
    print(model.score(X, y))
    print(model.score(X_test, y_test))

    gammas = np.linspace(1, int(gammas), int(gammas))
    # tds = np.arange(0.05, int(np.sqrt(X.shape[1])),0.05)
    tds = np.arange(0.05, 1.05, 0.05)

    results_grid_VD = grid_search_VD_v3(
        gammas, tds, X, X_train, y, feature_names, model,method="SHAP", dataset=dataset)
    pickle.dump(results_grid_VD, open("results_exp/"+dataset+"_VD.p", "wb"))

    # mean_precs, mean_cov, mean_dist_exp, empty_VDs, hp_values, std_precs, std_covs, std_dist_exp = results_grid_VD
    # best = best_hp_VD(mean_precs, mean_cov, mean_dist_exp, empty_VDs)
    # print("Best score :")
    # print(best)
    # print("hp values", hp_values[best])
    # print("Avg Precision", mean_precs[best], "+/-", std_precs[best])
    # print("Avg Coverage", mean_cov[best], "+/-", std_covs[best])
    # print("Avg Max Dist Exp", mean_dist_exp[best], "+/-", std_dist_exp[best])
    # print("Nb empty VD", empty_VDs[best])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset', help='Name of the selected dataset', default="flame")
    parser.add_argument(
        '-m', '--model', help='Name of the selected model', default="m0")
    parser.add_argument('-g', '--gammas', type=int,
                        help='Maximum value to test for gamma hp', default=20)
    parser.add_argument('-t', '--tds', type=int,
                            help='Maximum value to test for td hp', default=2)
    
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    gammas = args.gammas
    tds = args.tds

    main(dataset,model,gammas,tds)

