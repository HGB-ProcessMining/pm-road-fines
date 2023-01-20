import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


def prep_data_for_ml(time_start, index):
    """
    preprocess data for ml: calc duration and fill nans
    save data to csv
    time_start: dataframe with start time of each case
    index: index of data to preprocess
    """
    df = pd.read_csv(os.path.join(data_dir,"X_data", f'{index}full_prep0_data.csv')).set_index("case:concept:name")
    df["time:timestamp"] = pd.to_datetime(df['time:timestamp'], utc=True)
    #merge timestart to df by index
    df = df.merge(time_start, on="case:concept:name")

    df["duration"] = df["time:timestamp"] - df["time_start"]
    #duration in seconds
    df["duration"] = df["duration"].dt.total_seconds()
    df["duration"] = df["duration"] / 3600
    df["duration_norm"] = df["duration"] / df["duration"].max()
    df["duration_norm"].fillna(0, inplace=True)
    #fill nans of column expens with 0
    df["expense"].fillna(0, inplace=True)
    #fill nans of column points with 0
    df["points"].fillna(0, inplace=True)
    #fill nans of column amount with 0
    df["amount"].fillna(0, inplace=True)
    #fill nans of column notification_type with "None"
    df["notificationType"].fillna("None", inplace=True)
    #fill nans of column lastSent with "None"
    df["lastSent"].fillna("None", inplace=True)
    #fill nans of column paymentAmount with 0
    df["paymentAmount"].fillna(0, inplace=True)
    #fill nans of column matricola with "None"
    df["matricola"].fillna("None", inplace=True)
    #to csv, also sace index
    df.to_csv(os.path.join(data_dir, "X_data", f'{index}full_prep_data.csv'))

def apply(data_dir):
    """
    apply preprocessing to all data
    data_dir: directory where data is stored
    """
    df = pd.read_csv(os.path.join(data_dir,"X_data", f'{0}full_prep0_data.csv')).set_index("case:concept:name")
    time_start = pd.to_datetime(df['time:timestamp'], utc=True)
    #set column name
    time_start.name = 'time_start'

    for index in range(0, 9):
        print(f"preprocessing {index}")
        prep_data_for_ml(time_start, index)

def get_length_of_traces(data_dir):
    """
    get length of traces and store to csv
    """
    df = pd.read_csv(os.path.join(data_dir,"logs", f'log.csv')).set_index("case:concept:name")
    df["time:timestamp"] = pd.to_datetime(df['time:timestamp'], utc=True)
    df = df.groupby("case:concept:name").agg({"time:timestamp": "count"})
    #rename column
    df.rename(columns={"time:timestamp": "length"}, inplace=True)
    #save
    df.to_csv(os.path.join(data_dir, "logs", f'log_length.csv'))
    return df

def drop_out_traces(data_dir):
    """
    drop out traces in each dataset, which are already finished
    """
    trace_lengths = pd.read_csv(os.path.join(data_dir, "logs", f'log_length.csv')).set_index("case:concept:name")
    print(trace_lengths)
    #go through each file in X_data
    for index in range(0, 9):
        print(f"dropping {index}")
        df = pd.read_csv(os.path.join(data_dir,"X_data", f'{index}full_data.csv')).set_index("case:concept:name")
        #join df with trace_lengths
        df = df.join(trace_lengths, on="case:concept:name")
        #if value in column length is higher than index, drop column
        print(index)
        df = df[df["length"] >= index]
        df.to_csv(os.path.join(data_dir, "X_data", f'{index}full_prep0_data.csv'))


if __name__ == "__main__":
    data_dir = os.path.join('data')
    result_dir = os.path.join("results")
    #apply(data_dir)
    #df = get_length_of_traces(data_dir)
    #count number of traces with length of 9
    #print(df[df["length"] >= 6].count())
    #drop_out_traces(data_dir)
