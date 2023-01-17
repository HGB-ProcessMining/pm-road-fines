
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

def probs_to_log(log, probs, dataset_index):
    """
    add probabilities to log
    log: log with all cases
    probs: probabilities of cases
    dataset_index: index of dataset
    return: log with probabilities
    """
    #put new empty column probs in log
    for index, row in probs.iterrows():
        try:
            value = float(probs.loc[index, "PAY"])
            log.loc[index, "probs"][dataset_index] = value
        except: raise Exception(f"some problems with index {index}")

    return log

def probs_to_log_all_models(data_dir, result_dir, dataset_indices):
    """
    add probabilities to log for all models
    data_dir: path to data
    result_dir: path to results
    dataset_indices: indices of datasets
    return: log with probabilities
    """
    #create new empty column probs
    log = pd.read_csv(os.path.join(data_dir, "logs", "log.csv")).set_index("case:concept:name")
    log["probs"] = 0.0
    for ds_id in dataset_indices:
        probs = pd.read_csv(os.path.join(result_dir, "probabilities", f"{ds_id}_probabilities_test.csv")).set_index("case:concept:name")
        log = probs_to_log(log, probs, ds_id)

    #slice only the test indices
    test_indices = pd.read_csv(os.path.join(data_dir, "test_indices", "test_indices.csv"))
    log.to_csv(os.path.join(result_dir, "logs", f"log_{dataset_indices}_whole_log.csv"))
    log = log.loc[test_indices["case:concept:name"],:]
    #save log
    log.to_csv(os.path.join(result_dir, "logs", f"log_{dataset_indices}_test_data.csv"))
    return log

def apply():
    """
    apply function
    """
    data_dir = "./data"
    result_dir = "./results"

    probs_to_log_all_models(data_dir, result_dir, [0])

if __name__ == "__main__":
    apply()




