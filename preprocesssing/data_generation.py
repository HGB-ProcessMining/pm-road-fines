import pandas as pd
import numpy as np

def generate(filepath):
    """Generate the datasets"""
    df = pd.read_csv(filepath, parse_dates=["time:timestamp"], index_col=0)
    columns = df.columns
    grouped_df = df.groupby(['case:concept:name'])
    run_length = []
    for idx, group in grouped_df:
        run_length.append(len(group))

    longes_run = np.max(run_length)

    for number in range(longes_run):
        #the first dataset is already generated
        if number in [0,1]:
            continue
        print("CREATING DATAFRAME NR:", number)
        generate_current_state(number, grouped_df, columns=columns)


def generate_current_state(current_number, grouped_df, columns):
    """Generate the dataset for the current numbers of events, which where triggered."""
    file_name = f"/Users/sophie/Documents/DSE/WS_22_23/PRM/pm-road-fines/data/{current_number}_data.csv"
    data = []
    # cols: ['amount', 'org:resource', 'dismissal', 'concept:name', 'vehicleClass',
    # 'totalPaymentAmount', 'lifecycle:transition', 'time:timestamp',
    # 'article', 'points', 'case:concept:name', 'expense', 'notificationType',
    # 'lastSent', 'paymentAmount', 'matricola']
    for idx, group in grouped_df:
        print("group: ", idx)
        if current_number == 0:
            entry = group.iloc[current_number].values
            data.append(entry)
            continue

        pre_df = pd.DataFrame(columns=columns.to_list()).set_index('case:concept:name')
        key = group.iloc[0]['case:concept:name']
        value = list(group.iloc[0].values)
        value.remove(key)
        pre_df.loc[key] = value

        inter_number = current_number if current_number < len(group) else len(group)
        for row_idx in range(inter_number):
            current_df = pd.DataFrame(columns=columns.to_list()).set_index('case:concept:name')
            key = group.iloc[row_idx]['case:concept:name']
            value = list(group.iloc[row_idx].values)
            value.remove(key)
            current_df.loc[key] = value

            pre_df = pre_df.combine_first(current_df)

        entry = pre_df.loc[key].values
        data.append(entry)

    new_df = pd.DataFrame(data)
    new_df.to_csv(file_name)


if __name__ == "__main__":
    filepath = "/Users/sophie/Documents/DSE/WS_22_23/PRM/pm-road-fines/data/Road_Traffic_Fine_Management_Process_cleaned.csv"
    generate(filepath)