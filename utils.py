import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
import shap
from sklearn.datasets import load_iris, load_wine, load_digits
from tqdm import tqdm
from anchor import anchor_tabular
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import pickle
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_fscore_support
from time import time
from MAPLE import MAPLE
from lime.lime_tabular import LimeTabularExplainer
import multiprocessing
# from multiprocessing import Pool
import os
from multiprocessing.pool import ThreadPool as Pool
from tensorflow import keras
from tensorflow.keras import layers

def load_dataset(name, test_size=0.2):
    if "flame" in name :
        data=pd.read_csv("data/flame.txt",sep="\t",header=None)
        labels = data.iloc[:,-1]
        data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
        X=np.array(data)
        y=np.array(labels)-1
        feature_names=["x0","x1"]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        if 'mini' in name:
            np.random.seed(0)
            samples = np.random.randint(0,len(X),10)
            X = X[samples]
            y = y[samples]

    if "breast" == name :
        data=pd.read_csv("data/breast_tissue.txt",sep="\t")
        labels = data.iloc[:,0]
        data.drop(data.columns[0], axis=1, inplace=True)
        X=np.array(data)
        y=np.array(labels)
        enc = OrdinalEncoder()
        y = enc.fit_transform(y.reshape(-1, 1))
        feature_names=data.columns
        feature_names = [f.replace(' ','_') for f in feature_names]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if "breast_c" == name :
        #https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra
        data=pd.read_csv("data/breast_cancer.csv")
        labels = data.iloc[:,-1]
        data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
        X=np.array(data)
        y=np.array(labels)
        enc = OrdinalEncoder()
        y = enc.fit_transform(y.reshape(-1, 1))
        feature_names=data.columns
        feature_names = [f.replace(' ','_') for f in feature_names]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    elif "loans" in name:
        data = pd.read_csv("data/loans/loan_data.csv")
        labels = data["not.fully.paid"]
        data.drop(np.where(data.columns=="not.fully.paid")[0][0])
        X=np.array(data)
        y=np.array(labels)
        feature_names = list(data.columns)
        enc = OrdinalEncoder()
        X = enc.fit_transform(X)
        y = enc.fit_transform(y.reshape(-1, 1))
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        if 'mini' in name:
            np.random.seed(0)
            samples = np.random.randint(0,len(X),300)
            X = X[samples]
            y = y[samples]
    elif 'adult' in name :
        data = pd.read_csv("data/adult/adult.data",sep=",",header=None)
        labels = data.iloc[:,-1]
        data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
        X=np.array(data)
        y=np.array(labels)
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                                "Education-Num", "Marital_Status", "Occupation",
                                "Relationship", "Race", "Sex", "Capital_Gain",
                                "Capital_Loss", "Hours_per_week", "Country"]
        enc = OrdinalEncoder()
        X = enc.fit_transform(X)
        y = enc.fit_transform(y.reshape(-1, 1))
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        if 'mini' in name:
            np.random.seed(0)
            samples = np.random.randint(0,len(X),1000)
            X = X[samples]
            y = y[samples]
    elif name == "jain":
        data=pd.read_csv("data/jain.txt",sep="\t",header=None)
        labels = data.iloc[:,-1]
        data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
        X=np.array(data)
        y=np.array(labels)-1
        feature_names=["x0","x1"]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif name == "iris":
        data = load_iris()
        feature_names = data.feature_names
        feature_names = [f.replace(' ','_') for f in feature_names]
        X = data.data
        y = data.target
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif name == "wine":
        data = load_wine()
        feature_names = data.feature_names
        feature_names = [f.replace(' ','_') for f in feature_names]
        X = data.data
        y = data.target
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    elif 'MNIST' in name:
        data = load_digits()
        feature_names = data.feature_names
        feature_names = [f.replace(' ','_') for f in feature_names]
        X = data.data
        y = data.target
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X = X.reshape(len(X),8,8,1)
        y = keras.utils.to_categorical(y, 10)
        if 'mini' in name:
            np.random.seed(0)
            samples = np.random.randint(0,len(X),1000)
            X = X[samples]
            y = y[samples]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return  X_train, X_test, y_train, y_test, X, y, feature_names

def train_model(X_train, y_train, model_name='m0'):
    if model_name == 'm0':
        model = MLPClassifier(hidden_layer_sizes=(40,30,20,10),random_state=1, max_iter=1000)
    if model_name == 'm1':
        model = MLPClassifier(hidden_layer_sizes=(150,100,50),random_state=1, max_iter=1000)
    if model_name == 'image':
        input_shape = (X_train[0].shape[0],X_train[0].shape[1],1)
        model = keras.Sequential(
        [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
        ]
        )
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print(model.summary())
    elif model_name == 'toy':
        model = MLPClassifier(hidden_layer_sizes=(30,20,10),random_state=1, max_iter=700)
    model.fit(X_train, y_train)
    return model

def get_explanations(method, model, X_train, feature_names):
    if method == "SHAP":
        explainer = shap.Explainer(model.predict,X_train,feature_names=feature_names)
        shap_values = explainer(X_train)
        E = shap_values.values
    elif method == "DeepSHAP":
        X_train_subset = X_train[np.random.choice(np.arange(len(X_train)), 100, replace=False)]
        explainer = shap.DeepExplainer(model,X_train_subset)
        # X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
        shap_values = explainer.shap_values(X_train)
        # print(np.asarray(shap_values).shape)
        E = np.asarray(shap_values).reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2],10)
    elif method == "MAPLE":
        explainer = MAPLE(X_train, model.predict(X_train), X_train, model.predict(X_train))
        E = [explainer.explain(x)['coefs'] for x in X_train]
    elif method == "LIME":
        # class_names = np.unique(y)
        explainer = LimeTabularExplainer(training_data=X_train, feature_names=feature_names, class_names=None, discretize_continuous=False,random_state=0)
        E = [reorder_attributes(dict(explainer.explain_instance(x, model.predict_proba).as_list()), feature_names) for x in X_train]
    return E, explainer

def reorder_attributes(att, feature_names):    
    return [att[f] for f in feature_names if f in att.keys()]

def validity_domains(X_train,feature_names,model,threshold_diff=0.1,gamma=50, method="SHAP"):
    E, explainer = get_explanations(method, model, X_train, feature_names)
    if method=="DeepSHAP":
        y_pred = np.argmax(model.predict(X_train), axis=1)
        E = [E[i][int(y_pred[i])] for i in range(len(E))]
        X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
        # scaler = MinMaxScaler()
        # dists_exp = {c : scaler.fit_transform(pairwise_distances(E[np.where(y_pred==c)], metric="cosine")) for c in np.unique(y)}
        dists_exp = pairwise_distances(E, metric="cosine")
    else:
        dists_exp = pairwise_distances(E)

    if threshold_diff=="auto":
        threshold_diff = 0.1 * X_train.shape[1]/2
        print("threshold_diff =",threshold_diff)
    VD = np.asarray(dists_exp<threshold_diff).astype(int)

    models_VD=[]
    for vd in tqdm(VD):
        clf = OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(vd)])
        models_VD.append(clf)
    return VD, models_VD, explainer

def isinanchorarray(df, exp):
    isin = np.ones(len(df))
    for n in exp.names():
        
        if len(n.split())==3:
            feat, cond, val = n.split()
            if cond == '<=':
                isin = np.logical_and(isin, df[feat] <= float(val))
            elif cond == '<':
                isin = np.logical_and(isin, df[feat] < float(val))
            elif cond == '>=':
                isin = np.logical_and(isin, df[feat] >= float(val))
            elif cond == '>':
                isin = np.logical_and(isin, df[feat] > float(val))
        else:
            try:
                val1, cond1, feat, cond2, val2 = n.split()
            except:
                print(n.split())
            if cond1 == '<=':
                isin = np.logical_and(isin, df[feat] <= float(val1))
            elif cond1 == '<':
                isin = np.logical_and(isin, df[feat] < float(val1))
            
            if cond2 == '>=':
                isin = np.logical_and(isin, df[feat] >= float(val2))
            elif cond2 == '>':
                isin = np.logical_and(isin, df[feat] > float(val2))

    return isin

def compute_precs_cov_anchors(X, X_train, feature_names, y, model, threshold_prec = 0.95):
    
    explainer = anchor_tabular.AnchorTabularExplainer(
    np.unique(y),
    feature_names,
    X_train,
    {})

    exps = [explainer.explain_instance(x, model.predict, threshold=threshold_prec) for x in tqdm(X_train)]
    precisions = [exp.precision() for exp in exps]
    cov_a = [exp.coverage() for exp in exps]

    df = pd.DataFrame(X, columns=feature_names)
    coverages = [np.count_nonzero(isinanchorarray(df,exp))/len(X) for exp in exps]
    df = pd.DataFrame(X_train, columns=feature_names)
    coverages_train = [np.count_nonzero(isinanchorarray(df,exp))/len(X_train) for exp in exps]
    
    return precisions, cov_a, coverages, coverages_train

def compute_precs_cov_VD(X, X_train, model, models_VD, replace_empty = True):
    precs = []
    coverages = []
    coverages_train = []
    for i in range(len(X_train)):
        m=models_VD[i]

        A_train = np.where(m.predict(X_train)==1)
        A = np.where(m.predict(X)==1)
        if np.asarray(A_train).size != 0:
            pred_i = model.predict(np.asarray(X_train[i]).reshape(1,-1))
            prec = np.sum(pred_i == model.predict(X[A]))/len(A[0])
            cov_train = len(A_train[0])/len(X_train)
            cov = len(A[0])/len(X)        
            precs.append(prec)
            coverages.append(cov)
            coverages_train.append(cov_train)
        else:
            if replace_empty:
                precs.append(0) 
                coverages.append(0)
                coverages_train.append(0)

    return precs, coverages, coverages_train

def SVM_prec_recall(X, X_train, models_VD, model, feature_names, threshold_diff):
    explainer = shap.Explainer(model.predict,X_train,feature_names=feature_names)
    shap_values_train = explainer(X_train)
    shap_values = explainer(X)
    # dists_exp = pairwise_distances(shap_values.values)
    # VD = np.asarray(dists_exp<threshold_diff).astype(int)
    precs = []
    recalls = []
    f1s = []
    supps = []
    for i in range(len(X_train)):
        m = models_VD[i]
        y_pred = np.asarray(m.predict(X)==1).astype(int)
        # y_true = shap_values_train.values[i]
        dists_exp = pairwise_distances(shap_values_train.values[i].reshape(1, -1),shap_values.values)
        y_true = np.asarray(dists_exp<threshold_diff).astype(int)[0]
        # print(y_pred)
        # print(y_true)
        prec, recall, f1, supp = precision_recall_fscore_support(y_true, y_pred)
        precs.append(prec)    
        recalls.append(recall)    
        f1s.append(f1)    
        supps.append(supp)    
    return precs, recalls, f1s, supps


def diff_VD(X, X_train, models_VD, explainer):
    dist_exp = []
    dist_exp_train = []
    dist_obs = []
    dist_obs_train = []

    shap_values = explainer(X)
    shap_values_train = explainer(X_train)
    empty_VD = 0
    for i in range(len(X_train)):
        m=models_VD[i]
        A = np.where(m.predict(X)==1)
        if np.asarray(A).size != 0:
            d = pairwise_distances(X[A])
            dist_obs.append(np.max(d))

            d_exp = pairwise_distances(shap_values.values[A])
            dist_exp.append(np.max(d_exp))

        A_train = np.where(m.predict(X_train)==1)
        if np.asarray(A_train).size != 0:
            d_train = pairwise_distances(X_train[A_train])
            dist_obs_train.append(np.max(d_train))

            d_exp_train = pairwise_distances(shap_values_train.values[A_train])
            dist_exp_train.append(np.max(d_exp_train))
        else:
            empty_VD+=1
    return dist_exp, dist_exp_train, dist_obs, dist_obs_train, empty_VD

def diff_anchors(X, X_train, y, feature_names, explainer, model, threshold_prec):
    dist_exp = []
    dist_exp_train = []
    dist_obs = []
    dist_obs_train = []

    shap_values = explainer(X)
    shap_values_train = explainer(X_train)

    explainer = anchor_tabular.AnchorTabularExplainer(
    np.unique(y),
    feature_names,
    X_train,
    {})

    exps = [explainer.explain_instance(x, model.predict, threshold=threshold_prec) for x in tqdm(X_train)]
    df = pd.DataFrame(X, columns=feature_names) 
    df_train = pd.DataFrame(X_train, columns=feature_names) 
    for exp in exps:
        A = np.where(isinanchorarray(df,exp))
        if np.asarray(A).size != 0:
            d = pairwise_distances(X[A])
            dist_obs.append(np.max(d))

            d_exp = pairwise_distances(shap_values.values[A])
            dist_exp.append(np.max(d_exp))

        A_train = np.where(isinanchorarray(df_train,exp))
        if np.asarray(A_train).size != 0:
            d_train = pairwise_distances(X_train[A_train])
            dist_obs_train.append(np.max(d_train))

            d_exp_train = pairwise_distances(shap_values_train.values[A_train])
            dist_exp_train.append(np.max(d_exp_train))
    return dist_exp, dist_exp_train, dist_obs, dist_obs_train

def explanation_VD(idx, X_test, X_train, VD, model, models_VD, explainer, feature_names):
    vd = VD[idx]
    if X_train.shape[1]==2:
        disp = DecisionBoundaryDisplay.from_estimator(models_VD[idx], X_train,eps=0.1, grid_resolution=500, alpha=0.4, response_method="predict", plot_method='contour')
        plt.scatter(X_train[:,0],X_train[:,1], c = model.predict(X_train), edgecolor='k',cmap='viridis')
        plt.scatter(X_train[:,0][np.where(vd)],X_train[:,1][np.where(vd)], edgecolor='k',cmap='winter')
        plt.scatter(X_test[:,0],X_test[:,1], c='white',edgecolor='k')
        plt.scatter(X_train[idx,0],X_train[idx,1], c='r', marker='X')
        # plt.title("Validity domain for X["+str(idx)+"]")
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        # disp.ax_.add_line(Line2D([0.64,0.64,1],[0,0.69,0.69],color='red',linewidth=2,linestyle='--'))
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        # plt.savefig('toy_ex/anchors_fail.pdf')
        plt.show()
    shap_values = explainer(X_train)

    m = models_VD[idx]
    A_train = np.where(m.predict(X_train)==1)
    print("Explanation")
    print(shap_values[idx])
    values = np.argsort(np.abs(shap_values[idx].values))
    a = np.asarray([[shap_values[idx].values[values[0]], 0],[shap_values[idx].values[values[1]], 1]])
    plt.scatter(a[:,0],a[:,1],s=100, c='k', marker='X')
    shap.plots.beeswarm(shap_values[A_train])
    # plt.savefig('toy_ex/beeswarm.pdf')
    shap.plots.waterfall(shap_values[idx])
    # plt.savefig('toy_ex/exp_shap.pdf')

    data_col = []
    feature_col = []
    exp_col = []
    for i in A_train[0]:
        for f in range(len(feature_names)):
            data_col.append(X_train[i][f])
            feature_col.append(feature_names[f])
            exp_col.append(shap_values.values[i][f])
    df = pd.DataFrame(np.asarray([data_col, feature_col, exp_col]).T, columns=["feature values", "feature names", "exp_col"])
    df["feature values"] = df["feature values"].astype('float')
    df["exp_col"] = df["exp_col"].astype('float')

    norm = plt.Normalize(df["exp_col"].min(), df["exp_col"].max())
    sm = plt.cm.ScalarMappable(cmap="PiYG", norm=norm)
    ax = sns.swarmplot(data=df, x="feature names", y="feature values", hue="exp_col", palette="PiYG")
    data_col = []
    feature_col = []
    exp_col = []
    for f in range(len(feature_names)):
        data_col.append(X_train[idx][f])
        feature_col.append(feature_names[f])
        exp_col.append(shap_values.values[idx][f])
    df_2 = pd.DataFrame(np.asarray([data_col, feature_col, exp_col]).T, columns=["feature values", "feature names", "exp_col"])
    df_2["feature values"] = df["feature values"].astype('float')
    df_2["exp_col"] = df["exp_col"].astype('float')
    sns.swarmplot(data=df_2, x="feature names", y="feature values", hue="exp_col", color="k", marker='X',size = 10)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)
    ax.text(1.02, -0.05, "SHAP values", ha="left", va="top", transform=ax.transAxes)
    # plt.savefig('toy_ex/inverted_beeswarm.pdf')
    plt.show()
    # ax = sns.swarmplot(data=pd.DataFrame(X_train[A_train],columns=feature_names))
    # ax.set(xlabel="Features", ylabel="Value", title="Data distribution in validity domain")

def prec_cov(dataset, threshold_diff, gamma, verbose = True):
    X_train, X_test, y_train, y_test, X, y, feature_names = load_dataset(dataset)
    model = train_model(X_train,y_train)
    if verbose:
        print("->Model")
        print("Accuracy",model.score(X,y))
        print("Accuracy test",model.score(X_test,y_test))

        print("->VD")
    VD, models_VD, _ = validity_domains(X_train,feature_names,model,threshold_diff=threshold_diff, gamma=gamma)
    precs,covs,covs_train = compute_precs_cov_VD(X, X_train, model, models_VD)

    avg_prec_VD = np.mean(precs)
    avg_cov_VD = np.mean(covs)
    avg_cov_train_VD = np.mean(covs_train)

    if verbose:
        print("prec","%.3f" % avg_prec_VD,
                "cov","%.3f" % avg_cov_VD,
                "covs_train","%.3f" % avg_cov_train_VD)
        
        print("->Anchors")
    precs, cov_a, covs, covs_train = compute_precs_cov_anchors(X, X_train, feature_names, y, model, threshold_prec = np.mean(precs))
    
    avg_prec_anchors = np.mean(precs)
    avg_cov_anchors = np.mean(covs)
    avg_cov_train_anchors = np.mean(covs_train)

    if verbose:
        print("prec","%.3f" %avg_prec_anchors,
                "cov_a","%.3f" %np.mean(cov_a),
                "cov","%.3f" %avg_cov_anchors,
                "covs_train","%.3f" %avg_cov_train_anchors)
        
    return avg_prec_VD, avg_cov_VD, avg_prec_anchors, avg_cov_anchors

def compute_VD_and_scores(X, X_train, model,threshold_diff, gamma, shap_values, shap_values_train, dists_exp, replace_empty=True, lim = 500):

    if threshold_diff=="auto":
        threshold_diff = 0.1 * X_train.shape[1]/2
        print("threshold_diff =",threshold_diff)
    VD = np.asarray(dists_exp<threshold_diff).astype(int)

    if len(X_train)>lim:
        # print("Training set too large, computing on",lim," samples.")
        idx = np.random.choice(np.arange(len(X_train)),lim, replace=False)
    else:
        idx = np.arange(len(X_train))

    # models_VD=[OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(vd)]) for vd in VD]
    # print("Training SVM")
    # def f(id):
    #     return OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(VD[id])])
    
    # with Pool(multiprocessing.cpu_count()) as p :
    #     models_VD = p.map(f, idx)
    models_VD=[OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(VD[i])]) for i in tqdm(idx)]
    y_pred = model.predict(X_train)

    precs = []
    coverages = []
    coverages_train = []
    dist_exp = []
    dist_exp_train = []
    dist_obs = []
    dist_obs_train = []

    empty_VD = 0
    for j,i in enumerate(idx):
    # for i in idx:
        m = models_VD[j]
        # m = OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(VD[i])])

        A_train = np.where(m.predict(X_train)==1)
        A = np.where(m.predict(X)==1)
        # if np.asarray(A).size != 0:
            # d = pairwise_distances(X[A])
            # dist_obs.append(np.max(d))

            # d_exp = pairwise_distances(shap_values.values[A])
            # dist_exp.append(np.max(d_exp))

        if np.asarray(A_train).size != 0:
            # d_train = pairwise_distances(X_train[A_train])
            # dist_obs_train.append(np.max(d_train))

            # d_exp_train = pairwise_distances(shap_values_train.values[A_train])
            # dist_exp_train.append(np.max(d_exp_train))

            pred_i = y_pred[i]
            prec = np.sum(pred_i == model.predict(X[A]))/len(A[0])
            cov_train = len(A_train[0])/len(X_train)
            cov = len(A[0])/len(X)        
            precs.append(prec)
            coverages.append(cov)
            coverages_train.append(cov_train)
        else:
            empty_VD+=1
            if replace_empty:
                precs.append(0) 
                coverages.append(0)
                coverages_train.append(0)

    return precs, coverages, dist_exp, empty_VD

def compute_VD_and_scores_v2(X, X_train, y, model,threshold_diff, gamma, dists_exp, max_dist, replace_empty=True, lim = 500):
    if threshold_diff=="auto":
        threshold_diff = 0.1 * X_train.shape[1]/2
        print("threshold_diff =",threshold_diff)
    VD = {c : np.asarray(dists_exp[c]<threshold_diff).astype(int) for c in np.unique(y)}
    y_pred = model.predict(X_train)
    pred_idx = {c : np.where(y_pred==c) for c in np.unique(y)}
    # VD = np.asarray(dists_exp<threshold_diff).astype(int)

    if len(X_train)>lim:
        idx = np.random.choice(np.arange(len(X_train)),lim, replace=False)
    else:
        idx = np.arange(len(X_train))

    # models_VD=[OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[np.where(VD[i])]) for i in tqdm(idx)]
    # models_data = [X_train[pred_idx[y_pred[i]]][np.where(VD[y_pred[i]][np.where(pred_idx[y_pred[i]] == i)])]) for i in tqdm(idx)]
    # models_VD=[OneClassSVM(nu=0.01,gamma=gamma).fit(X_train[pred_idx[y_pred[i]]][np.where(VD[y_pred[i]][np.where(pred_idx[y_pred[i]] == i)])]) for i in tqdm(idx)]
    models_VD = []
    # for c in np.unique(y):
    #     X_train_c = X_train[pred_idx[c]]
    #     VD_c = VD[c]
    #     for i in range(len(VD_c)):
    #         models_VD.append(OneClassSVM(nu=0.01,gamma=gamma).fit(X_train_c[np.where(VD_c[i])]))
    for i in idx:
        c = y_pred[i]
        X_train_c = X_train[pred_idx[c]]
        VD_c = VD[c]
        idx_vd = np.where(pred_idx[c]==i)[0][0]
        models_VD.append(OneClassSVM(nu=0.01,gamma=gamma).fit(X_train_c[np.where(VD_c[idx_vd])]))
    precs = []
    coverages = []
    coverages_train = []
    dist_exp = []
    mdos = []


    empty_VD = 0
    for j,i in enumerate(idx):
        m = models_VD[j]

        A_train = np.where(m.predict(X_train)==1)
        A = np.where(m.predict(X)==1)
        if np.asarray(A_train).size != 0:
            pred_i = y_pred[i]
            prec = np.sum(pred_i == model.predict(X[A]))/len(A[0])
            # print(i,j)
            # print(pred_i)
            # print(X[A])
            # print(model.predict(X[A]))
            # print(len(A[0]))
            # print("--------------------------------")
            # print(prec)
            # print("--------------------------------")

            cov_train = len(A_train[0])/len(X_train)
            cov = len(A[0])/len(X)        
            mdo = np.max(pairwise_distances(X[A]))/max_dist
            precs.append(prec)
            coverages.append(cov)
            coverages_train.append(cov_train)
            mdos.append(mdo)
        else:
            empty_VD+=1
            if replace_empty:
                precs.append(0) 
                coverages.append(0)
                coverages_train.append(0)
                mdos.append(0)

    return precs, coverages, dist_exp, empty_VD, mdos

def grid_search_VD(gammas, tds, X, X_train, feature_names, model, method = "SHAP"):
    mean_precs_df = []
    mean_cov_df = []
    mean_dist_exp_df = []
    empty_VDs_df = []
    hp_values = []

    # E, explainer = get_explanations(method, model, X_train, feature_names)
    explainer = shap.Explainer(model.predict,X_train,feature_names=feature_names)
    shap_values = explainer(X)
    shap_values_train = explainer(X_train)
    dists_exp = pairwise_distances(shap_values_train.values)

    std_precs = []
    std_covs = []
    std_dist_exp = []

    with tqdm(total=len(gammas) * len(tds)) as pbar:
        for gamma in gammas:
            mean_precs = []
            mean_cov = []
            mean_dist_exp = []
            empty_VDs = []
            
            for td in tds:
                precs, covs, dist_exp, empty_VD = compute_VD_and_scores(X, X_train, model, td, gamma, shap_values, shap_values_train, dists_exp, replace_empty=True)
                mean_precs.append(np.mean(precs))
                mean_cov.append(np.mean(covs))
                mean_dist_exp.append(np.mean(dist_exp))

                std_precs.append(np.std(precs))
                std_covs.append(np.std(covs))
                std_dist_exp.append(np.std(dist_exp))

                empty_VDs.append(empty_VD)
                hp_values.append((gamma,td))
                pbar.update(1)
            mean_precs_df.append(mean_precs)
            mean_cov_df.append(mean_cov)
            mean_dist_exp_df.append(mean_dist_exp)
            empty_VDs_df.append(empty_VDs)


    # ax = sns.heatmap(pd.DataFrame(mean_precs_df),xticklabels=tds,yticklabels=gammas, annot=True, cbar=False)
    # ax.set(xlabel="Distance threshold", ylabel="Gamma", title="Precision")
    # plt.show()
    # ax = sns.heatmap(pd.DataFrame(mean_cov_df),xticklabels=tds,yticklabels=gammas, annot=True, cbar=False)
    # ax.set(xlabel="Distance threshold", ylabel="Gamma", title="Coverage")
    # plt.show()
    # ax = sns.heatmap(pd.DataFrame(mean_dist_exp_df),xticklabels=tds,yticklabels=gammas, annot=True, cbar=False)
    # ax.set(xlabel="Distance threshold", ylabel="Gamma", title="Average max distance between explanations")
    # plt.show()
    # ax = sns.heatmap(pd.DataFrame(empty_VDs_df),xticklabels=tds,yticklabels=gammas, annot=True, cbar=False)
    # ax.set(xlabel="Distance threshold", ylabel="Gamma", title="empty VD")
    # plt.show()

    mean_precs = np.asarray(mean_precs_df).flatten().tolist()
    mean_cov = np.asarray(mean_cov_df).flatten().tolist()
    mean_dist_exp = np.asarray(mean_dist_exp_df).flatten().tolist()
    empty_VDs = np.asarray(empty_VDs_df).flatten().tolist()

    return mean_precs, mean_cov, mean_dist_exp, empty_VDs, hp_values, std_precs, std_covs, std_dist_exp

def grid_search_VD_v2(gammas, tds, X, X_train, y, feature_names, model, method = "SHAP", dataset = "adult", metric = "euclidean"):
    if method == "SHAP":
        path = "results_exp/shap_values/"+dataset+".p"
        if os.path.isfile(path):
            print("Getting explanations")
            shap_values = pickle.load(open(path, "rb"))
        else:
            print("Producing explanations")
            explainer = shap.Explainer(model.predict_proba,X_train,feature_names=feature_names)
            shap_values = explainer(X)

            # exp = [shap_values.values[i][:,y[i]] for i in range(len(X))]
            # shap_values = np.asarray(exp)

            # scaler = MinMaxScaler()
            # scaled_shap_values = np.asarray([scaler.fit_transform(sv.reshape(-1, 1)) for sv in shap_values.values])
            # shap_values = scaled_shap_values.reshape(scaled_shap_values.shape[0],scaled_shap_values.shape[1])
            # shap_values = shap_values.values
            shap_values = np.asarray(shap_values.values)
            # shap_values = np.reshape(shap_values, (shap_values.shape[0],shap_values.shape[1]*shap_values.shape[2]))
            pickle.dump(shap_values, open(path, "wb"))

        id_train = [np.where(np.all(X==X_train[i], axis=1))[0][0] for i in range(len(X_train))]
        shap_values_train = shap_values[id_train]
        # dists_exp = pairwise_distances(shap_values_train, metric=metric)
        # dists_exp = {c : pairwise_distances(shap_values_train, metric=metric) for c in np.unique(y)}
        dists_exp = pairwise_distances(shap_values_train, metric=metric)
    results_dict = {'hp':[],
                     'precisions':[],
                     'coverages':[],
                     'dist exp':[],
                     'empty VDs':[]}
    print("Grid search")
    path = "results_exp/"+dataset+"_VD.p"
    if os.path.isfile(path):
        print("Recovering previous results")
        results_dict = pickle.load(open(path, "rb"))

    with tqdm(total=len(gammas) * len(tds)) as pbar:
        for td in tds:
            for gamma in gammas:     
                print(gamma,td)
                if (gamma,td) not in results_dict['hp']:
                    precs, covs, dist_exp, empty_VD = compute_VD_and_scores(X, X_train, model, td, gamma, shap_values, shap_values_train, dists_exp, replace_empty=True)
                    results_dict['precisions'].append(precs)
                    results_dict['coverages'].append(covs)
                    results_dict['dist exp'].append(dist_exp)
                    results_dict['empty VDs'].append(empty_VD)
                    results_dict['hp'].append((gamma,td))
                    pickle.dump(results_dict, open(path, "wb"))
                pbar.update(1)
    return results_dict

def grid_search_VD_v3(gammas, tds, X, X_train, y, feature_names, model, method = "SHAP", dataset = "adult", metric = "euclidean"):
    if method == "SHAP":
        path = "results_exp/shap_values/"+dataset+".p"
        if os.path.isfile(path):
            print("Getting explanations")
            shap_values = pickle.load(open(path, "rb"))
        else:
            print("Producing explanations")
            explainer = shap.Explainer(model.predict,X_train,feature_names=feature_names)
            shap_values = explainer(X_train)

            shap_values = np.asarray(shap_values.values)
            pickle.dump(shap_values, open(path, "wb"))
        scaler = MinMaxScaler()
        y_pred = model.predict(X_train)
        dists_exp = {c : scaler.fit_transform(pairwise_distances(shap_values[np.where(y_pred==c)], metric=metric)) for c in np.unique(y)}
    results_dict = {'hp':[],
                     'precisions':[],
                     'coverages':[],
                     'dist exp':[],
                     'empty VDs':[],
                     'mdos':[]}
    print("Grid search")
    path = "results_exp/"+dataset+"_VD.p"
    if os.path.isfile(path):
        print("Recovering previous results")
        results_dict = pickle.load(open(path, "rb"))
    max_dist = np.max(pairwise_distances(X))
    with tqdm(total=len(gammas) * len(tds)) as pbar:
        for td in tds:
            for gamma in gammas:     
                print(gamma,td)
                if (gamma,td) not in results_dict['hp']:
                    precs, covs, dist_exp, empty_VD, mdos = compute_VD_and_scores_v2(X, X_train, y, model, td, gamma, dists_exp, max_dist,replace_empty=True)
                    results_dict['precisions'].append(precs)
                    results_dict['coverages'].append(covs)
                    results_dict['dist exp'].append(dist_exp)
                    results_dict['empty VDs'].append(empty_VD)
                    results_dict['hp'].append((gamma,td))
                    results_dict['mdos'].append(mdos)
                    pickle.dump(results_dict, open(path, "wb"))
                pbar.update(1)
    return results_dict

def compute_anchors_and_scores(X, X_train, feature_names, model, delta, tau, B, threshold_prec, t, shap_values, explainer):
    np.random.seed(int(t))
    exps = [explainer.explain_instance(x, model.predict, threshold=threshold_prec, delta=delta, tau=tau, batch_size=B) for x in tqdm(X_train)]
    precisions = [exp.precision() for exp in exps]

    df = pd.DataFrame(X, columns=feature_names)
    coverages = [np.count_nonzero(isinanchorarray(df,exp))/len(X) for exp in exps]

    dist_exp = []
    for exp in exps:
        A = np.where(isinanchorarray(df,exp))
        if np.asarray(A).size != 0:

            d_exp = pairwise_distances(shap_values.values[A]) #TODO can make it faster
            dist_exp.append(np.max(d_exp))
    return precisions, coverages, dist_exp

def compute_anchors_and_scores_v2(X, X_train, feature_names, model, delta, tau, B, threshold_prec, t, explainer, max_dist, lim = 1000):
    print(delta, tau, B, t)
    if len(X_train)>lim:
        # print("Training set too large, computing on",lim," samples.")
        idx = np.random.choice(np.arange(len(X_train)),lim, replace=False)
    else:
        idx = np.arange(len(X_train))
    np.random.seed(int(t))
    exps = [explainer.explain_instance(x, model.predict, threshold=threshold_prec, delta=delta, tau=tau, batch_size=B) for x in tqdm(X_train[idx])]
    precisions = [exp.precision() for exp in exps]

    df = pd.DataFrame(X, columns=feature_names)
    anchors_obs = [isinanchorarray(df,exp) for exp in exps]
    coverages = [np.count_nonzero(a_o)/len(X) for a_o in anchors_obs]
    mdos = [np.max(pairwise_distances(X[a_o]))/max_dist if len(X[a_o])>1 else 0 for a_o in anchors_obs]

    return precisions, coverages, mdos

def grid_search_anchors(X, y, X_train, feature_names, model, deltas, taus, Bs, threshold_prec, tries):
    shap_explainer = shap.Explainer(model.predict,X_train,feature_names=feature_names)
    shap_values = shap_explainer(X)

    explainer = anchor_tabular.AnchorTabularExplainer(
    np.unique(y),
    feature_names,
    X_train,
    {})

    hp_values = []
    mean_precs = []
    mean_cov = []
    mean_dist_exp = []

    std_precs = []
    std_covs = []
    std_dist_exp = []

    with tqdm(total=len(deltas) * len(taus) * len(Bs) * len(tries)) as pbar:
        for delta in deltas:
            for tau in taus:
                for B in Bs:
                    for t in tries:
                        precs, covs, dist_exp = compute_anchors_and_scores(X, X_train, feature_names, model, delta, tau, B, threshold_prec, t, shap_values, explainer)
                        hp_values.append((delta,tau,B,t))
                        mean_precs.append(np.mean(precs))
                        mean_cov.append(np.mean(covs))
                        mean_dist_exp.append(np.mean(dist_exp))

                        std_precs.append(np.std(precs))
                        std_covs.append(np.std(covs))
                        std_dist_exp.append(np.std(dist_exp))

                        pbar.update(1)

    return mean_precs, mean_cov, mean_dist_exp, hp_values, std_precs, std_covs, std_dist_exp

def grid_search_anchors_v2(X, y, X_train, feature_names, model, deltas, taus, Bs, threshold_prec, tries):

    explainer = anchor_tabular.AnchorTabularExplainer(
    np.unique(y),
    feature_names,
    X_train,
    {})
    max_dist = np.max(pairwise_distances(X))
    hps = []    
    for delta in deltas:
        for tau in taus:
            for B in Bs:
                for t in tries:
                    hps.append([delta,tau,B,t])

    results_dict = {'hp':[],
                     'precisions':[],
                     'coverages':[],
                     'mdos':[]}
    def f(hp):
       delta, tau, B, t = hp
       precisions, coverages, mdos = compute_anchors_and_scores_v2(X, X_train, feature_names, model, delta, tau, B, threshold_prec, t, explainer, max_dist)
       return hp, precisions, coverages, mdos

    with Pool(multiprocessing.cpu_count()) as p :
        results = p.map(f, hps)
        # print(results)
        for r in results:
            results_dict['hp'].append(r[0])
            results_dict['precisions'].append(r[1])
            results_dict['coverages'].append(r[2])
            results_dict['mdos'].append(r[3])
    
    return results_dict

def grid_search_anchors_v3(X, y, X_train, dataset, feature_names, model, deltas, taus, Bs, threshold_prec, tries):

    explainer = anchor_tabular.AnchorTabularExplainer(
    np.unique(y),
    feature_names,
    X_train,
    {})
    max_dist = np.max(pairwise_distances(X))

    results_dict = {'hp':[],
                     'precisions':[],
                     'coverages':[],
                     'mdos':[]}
    
    print("Grid search")
    path = "results_exp/"+dataset+"_anchors.p"
    if os.path.isfile(path):
        print("Recovering previous results")
        results_dict = pickle.load(open(path, "rb"))

    with tqdm(total=len(deltas) * len(taus) * len(Bs) * len(tries)) as pbar:
        for delta in deltas:
            for tau in taus:
                for B in Bs:
                    for t in tries:

                        print(delta,tau, B, t)
                        if (delta,tau, B, t) not in results_dict['hp']:
                            precisions, coverages, mdos = compute_anchors_and_scores_v2(X, X_train, feature_names, model, delta, tau, B, threshold_prec, t, explainer, max_dist)
                            results_dict['precisions'].append(precisions)
                            results_dict['coverages'].append(coverages)
                            results_dict['mdos'].append(mdos)
                            results_dict['hp'].append((delta,tau, B, t))
                            pickle.dump(results_dict, open(path, "wb"))
                        pbar.update(1)

    return results_dict


def get_local_exp(x, explainer, model, label="pred", mode="classification"):
    if mode=="regression":
        label = "all"
    if label=="pred":
        label=int(model.predict(x.reshape(1, -1)))
    if label=="all":
        return explainer.shap_values(x,nsamples="auto",l1_reg="auto")
    return explainer.shap_values(x,nsamples="auto",l1_reg="auto")[label]

def compute_all_exp(X, explainer, model, label):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = []
    for ii in tqdm(range(xx.shape[0])):
        for j in range(xx.shape[1]):
            x = np.array([xx[ii][j],yy[ii][j]])
            e_x = get_local_exp(x, explainer, model, label=label)
            Z.append(e_x)
    return Z

def diff_explanation(e_x,e_i, metric='euclidean'):
    return pairwise_distances(np.reshape(e_x,(1,-1)),np.reshape(e_i,(1,-1)),metric) if metric in distance_metrics() else 0

from sklearn.metrics.pairwise import distance_metrics

def toy_ex_figures(X, y, feature_names, explanations, model, explainer, i, p1, p2, p3, metric='euclidean'):
    exps = explanations.copy()
    x_i = X[i]
    x_p1 = X[p1]
    x_p2 = X[p2]
    x_p3 = X[p3]
    label = int(model.predict(x_i.reshape(1, -1)))
    print(f"x_i:{x_i}, label:{label}")
    e_i = get_local_exp(x_i, explainer, model, label)
    e_p1 = get_local_exp(x_p1, explainer, model, label)
    e_p2 = get_local_exp(x_p2, explainer, model, label)
    e_p3 = get_local_exp(x_p3, explainer, model, label)

    print("e_i (red)",e_i)
    print("e_p1 (blue)",e_p1)
    print("e_p2 (magenta)",e_p2)
    print("e_p3 (lime)",e_p3)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = []
    for ii in tqdm(range(xx.shape[0])):
        for j in range(xx.shape[1]):
            e_x = exps.pop(0)[label]
            Z.append(diff_explanation(e_x,e_i, metric=metric))
    Z=np.asarray(Z)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.scatter(x_p1[0], x_p1[1],c='blue',marker='P')
    plt.scatter(x_p2[0], x_p2[1],c='magenta',marker='P')
    plt.scatter(x_p3[0], x_p3[1],c='lime',marker='P')
    plt.scatter(x_i[0], x_i[1],c='red',marker='X')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.colorbar()
    plt.savefig('toy_ex/validity.pdf')
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].barh(['sepal\nwidth\n(cm)', 'petal\nlength\n(cm)'],e_i,color='red')
    axs[0, 1].barh([" ",""],e_p1,color='blue')
    axs[1, 0].barh(['sepal\nwidth\n(cm)', 'petal\nlength\n(cm)'],e_p2,color='magenta')
    axs[1, 1].barh([" ",""],e_p3,color='lime')
    plt.savefig('toy_ex/explanations.pdf')
