from utils import load_dataset, train_model, grid_search_VD, grid_search_anchors
import numpy as np
import pickle

def best_hp_VD(mean_precs, mean_cov, mean_dist_exp, empty_VDs):
    return np.argmax(np.asarray(mean_precs)+np.asarray(mean_cov)-np.asarray(mean_dist_exp)-np.asarray(empty_VDs))

def best_hp_anchors(mean_precs, mean_cov):
    return np.argmax(np.asarray(mean_precs)+np.asarray(mean_cov))

dataset = "flame"
model_name = "m0"
X_train, X_test, y_train, y_test, X, y, feature_names = load_dataset(dataset)
model = train_model(X_train,y_train, model_name)
print(model.score(X, y))
print(model.score(X_test, y_test))

gammas = np.linspace(1, 20, 20)
tds = np.linspace(0.1, 2, 20)

results_grid_VD = grid_search_VD(gammas, tds, X, X_train, feature_names, model)
pickle.dump(results_grid_VD, open("results_exp/"+dataset+"_VD.p", "wb"))

mean_precs, mean_cov, mean_dist_exp, empty_VDs, hp_values, std_precs, std_covs, std_dist_exp = results_grid_VD
best = best_hp_VD(mean_precs, mean_cov, mean_dist_exp, empty_VDs)
print("Best score :")
print(best)
print("hp values",hp_values[best])
print("Avg Precision",mean_precs[best],"+/-",std_precs[best])
print("Avg Coverage",mean_cov[best],"+/-",std_covs[best])
print("Avg Max Dist Exp",mean_dist_exp[best],"+/-",std_dist_exp[best])
print("Nb empty VD",empty_VDs[best])

deltas = [0.05,0.1]
taus = [0.1,0.15]
Bs = [10,50,100]
threshold_prec = 1
tries = np.linspace(1, 5, 5)

results_grid_anchors = grid_search_anchors(X, y, X_train, feature_names, model, deltas, taus, Bs, threshold_prec, tries)
pickle.dump(results_grid_anchors, open("results_exp/"+dataset+"_anchors.p", "wb"))

mean_precs, mean_cov, mean_dist_exp, hp_values, std_precs, std_covs, std_dist_exp = results_grid_anchors
best = best_hp_anchors(mean_precs, mean_cov)
print("Best score :")
print(best)
print("hp values", hp_values[best])
print("Avg Precision",mean_precs[best],"+/-",std_precs[best])
print("Avg Coverage",mean_cov[best],"+/-",std_covs[best])
print("Avg Max Dist Exp",mean_dist_exp[best],"+/-",std_dist_exp[best])