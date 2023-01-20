import pandas as pd
import numpy as np

def generate():
    """Generate the datasets"""
    filepath = r"C:\Users\PhilippKastenhofer\repositories\pm-road-fines\data\logs\log.csv"
    df = pd.read_csv(filepath, parse_dates=["time:timestamp"], index_col=0)
    concept_names = df['case:concept:name'].unique()
    columns = list(df.columns)
    columns.remove('case:concept:name')

    filepath = "C:\\Users\\PhilippKastenhofer\\repositories\\pm-road-fines\\data\\new_data\\"
    for number in range(2,9):
        if number == 0:
            continue
        data_path = filepath + str(number) + "_data.csv"
        data_df = pd.read_csv(data_path, index_col=0)
        data_df.set_axis(columns, axis=1,inplace=True)
        data_df['case:concept:name'] = concept_names
        data_df.to_csv(filepath + str(number) + "full_data.csv")


if __name__ == "__main__":
    generate()