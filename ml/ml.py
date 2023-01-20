
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

"""important note:
PAY: 0
NOT PAY: 1
"""

def load_data(data_path, label_path):
    """
    load data and labels
    data_path: path to data_{index}.csv
    label_path: path to labels.csv
    """

    labels = pd.read_csv(label_path).set_index("case:concept:name")
    df = pd.read_csv(data_path).set_index("case:concept:name")
    #join X0 and y
    data = df.join(labels)
    return data

def calc_label_col(data, threshold=10):
    """
    calculate label column
    data: dataframe with data and labels
    threshold: threshold where fine is paid or not paid
    """
    data["cat_diff"] = pd.cut(data["diff"], bins=[-10000,threshold,10000], labels=[0,1])
    return data


def preprocess_data(data, use_scaler=True):
    """
    preprocess data: get dummies, extract features from timestamp, standar scale numeric features
    data: dataframe with data and labels
    use_scaler: if True, standar scale numeric features
    return: X, y    
    """

    numeric_features = ["amount", "points", "duration_norm", "totalPaymentAmount", "expense", "paymentAmount"]
    categorical_features = ["vehicleClass", "article", "dismissal", "concept:name","notificationType", "lastSent", "matricola"]
    timestamp_features = ["time:timestamp"]
    target = "cat_diff"

    X = data[numeric_features + categorical_features + timestamp_features]
    y = data[target]

    #get dummies of categorical features
    X = pd.get_dummies(X, columns=categorical_features)
    #convert timestamp to datetime
    X["time:timestamp"] = pd.to_datetime(X["time:timestamp"])

    #extract features from timestamp
    X["year"] = pd.to_datetime(X["time:timestamp"], utc=True).dt.year
    X["month"] = pd.to_datetime(X["time:timestamp"], utc=True).dt.month
    X["day"] = pd.to_datetime(X["time:timestamp"], utc=True).dt.day
    X["hour"] = pd.to_datetime(X["time:timestamp"], utc=True).dt.hour
    X["weekday"] = pd.to_datetime(X["time:timestamp"], utc=True).dt.weekday
    #drop timestamp
    X = X.drop("time:timestamp", axis=1)

    #standar scale numeric features
    if use_scaler:
        scaler = StandardScaler()
        X_ = scaler.fit_transform(X)

        #X to Dataframe with initial index
        X_ = pd.DataFrame(X_, index=X.index)
        #put feature names back
        X_.columns = X.columns.astype(str)
        X = X_
    return X, y


def apply_train_test_split(X,y, test_indices, on_test_indices):
    """
    split data into train and test
    X: dataframe with features
    y: dataframe with labels
    test_indices: path to test_indices.csv
    return: X_train, X_test, y_train, y_test
    """
    #split data into train and test by test_indices.csv
    if on_test_indices:
        print("split on test_indices")
        test_indices = pd.read_csv(test_indices)
        test_indices = test_indices["case:concept:name"].values
        X_train = X[~X.index.isin(test_indices)]
        X_test = X[X.index.isin(test_indices)]
        y_train = y[~y.index.isin(test_indices)]
        y_test = y[y.index.isin(test_indices)]
    else:
        #apply sklearn train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def evaluate_models(model, X_train, X_test, y_train, y_test, dataset_index):
    """
    evaluate model on train and test data and save metrics to dataframe and csv
    model: model to evaluate
    X_train: dataframe with features of train data
    X_test: dataframe with features of test data
    y_train: dataframe with labels of train data
    y_test: dataframe with labels of test data
    dataset_index: index of dataset
    return: dataframe with metrics
    """
    #apply on test data
    y_pred_test = model.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    #get also metrics on train data
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train)
    rec_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    #write results to dataframe --> columns train, test and rows metrics
    results = pd.DataFrame( columns=["train", "test"])
    results.loc["accuracy"] = [acc_train, acc_test]
    results.loc["precision"] = [prec_train, prec_test]
    results.loc["recall"] = [rec_train, rec_test]
    results.loc["f1"] = [f1_train, f1_test]

    #save results to csv
    results.to_csv(f"./results/model_metrics/{dataset_index}_metrics.csv")
    return results
    

def plot_confusion_matrix(model, X_train, X_test, y_train, y_test, dataset_index):
    """
    plot confusion matrix for train and test data and save figures
    model: model to evaluate
    X_train: dataframe with features of train data
    X_test: dataframe with features of test data
    y_train: dataframe with labels of train data
    y_test: dataframe with labels of test data
    dataset_index: index of dataset
    """
    #plot confusion matrix
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    cm_test = confusion_matrix(y_test, y_pred_test)

    df_cm_test = pd.DataFrame(cm_test, index = [i for i in ["PAY", "NOT PAY"]], columns = [i for i in ["PAY", "NOT PAY"]])
    df_cm_test_rel = df_cm_test.div(df_cm_test.sum(axis=1), axis=0)

    cm_train = confusion_matrix(y_train, y_pred_train)

    df_cm_train = pd.DataFrame(cm_train, index = [i for i in ["PAY", "NOT PAY"]], columns = [i for i in ["PAY", "NOT PAY"]])
    df_cm_train_rel = df_cm_train.div(df_cm_train.sum(axis=1), axis=0)

    #set labels
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    #plot df_cm, where colors should be based on relative frequencies
    sns.heatmap(df_cm_test_rel, annot=True, ax=ax[0],  cmap="Blues")
    sns.heatmap(df_cm_train_rel, annot=True, ax=ax[1],  cmap="Blues")
    ax[0].set_title("Test Data")
    ax[1].set_title("Train Data")

    plt.savefig(f"./results/confusion_matrix/{dataset_index}_confusion_matrix.png", dpi=600)

def get_probabilites(model, X_train, X_test,y_train,y_test, dataset_index):
    """
    get probabilities for each class in train and test and save to csv
    model: model to evaluate
    X_train: dataframe with features of train data
    X_test: dataframe with features of test data
    dataset_index: index of dataset
    return: y_pred_train, y_pred_test
    """

    #get probabilities for each class
    y_probs_test = model.predict_proba(X_test)
    y_probs_train = model.predict_proba(X_train)
    
    #write to dataframe
    df_y_probs_test = pd.DataFrame(y_probs_test, columns=["PAY", "NOT PAY"])
    df_y_probs_train = pd.DataFrame(y_probs_train, columns=["PAY", "NOT PAY"])

    #put indices back
    df_y_probs_test.index = X_test.index
    df_y_probs_train.index = X_train.index
    df_y_probs_train["true_label"] = y_train
    df_y_probs_test["true_label"] = y_test
    df_y_probs_train["predicted_label"] = model.predict(X_train)
    df_y_probs_test["predicted_label"] = model.predict(X_test)

    #save to csv
    df_y_probs_test.to_csv(f"./results/probabilities/{dataset_index}_probabilities_test.csv")
    df_y_probs_train.to_csv(f"./results/probabilities/{dataset_index}_probabilities_train.csv")

    return y_probs_train, y_probs_test

def get_feature_importance(model, X_train, dataset_index):
    """
    get feature importance of model and save to csv
    model: model to evaluate
    X_train: dataframe with features of train data
    dataset_index: index of dataset
    """
    #get feature importance
    feature_importance = model.feature_importances_
    df_feature_importance = pd.DataFrame(feature_importance, index=X_train.columns, columns=["importance"])
    df_feature_importance = df_feature_importance.sort_values(by="importance", ascending=False)
    #save to csv
    df_feature_importance.to_csv(f"./results/feature_importance/{dataset_index}_feature_importance.csv")

def apply(dataset_index):
    data_dir = "./data"
    data_file = f"X_data/{dataset_index}full_prep_data.csv"
    label_file = "labels/paid.csv"
    test_file = "test_indices/test_indices.csv"

    #make absolute directorys
    data_path = os.path.abspath(os.path.join(data_dir, data_file))
    label_path = os.path.abspath(os.path.join(data_dir, label_file))
    test_path = os.path.abspath(os.path.join(data_dir, test_file))

    #prepare data
    data = load_data(data_path, label_path)
    data = calc_label_col(data, threshold=2)
    X,y = preprocess_data(data)
    X_train, X_test, y_train, y_test = apply_train_test_split(X,y, test_path, on_test_indices=True)

    #apply random forest
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    #fit model
    model.fit(X_train, y_train)

    results = evaluate_models(model, X_train, X_test, y_train, y_test, dataset_index)
    plot_confusion_matrix(model, X_train, X_test, y_train, y_test, dataset_index)
    get_probabilites(model, X_train, X_test,y_train, y_test, dataset_index)
    get_feature_importance(model, X_train, dataset_index)
    return results


if __name__ == "__main__":
    for i in range(0,9):
        print("Applying model to dataset", i, "")
        print(apply(i))





    