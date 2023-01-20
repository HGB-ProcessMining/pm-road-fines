import pandas as pd
import numpy as np
import concurrent.futures
import itertools

def generate(filepath):
    """Generate the datasets"""
    df = pd.read_csv(filepath, parse_dates=["time:timestamp"], index_col=0)
    columns = df.columns
    grouped_df = df.groupby(['case:concept:name'])
    run_length = []
    for idx, group in grouped_df:
        run_length.append(len(group))

    longes_run = np.max(run_length)

    for number in range(2,longes_run):
        #the first dataset is already generated
        if number in [0,1]:
            continue
        print("CREATING DATAFRAME NR:", number)
        generate_current_state(number, grouped_df, columns=columns)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    # # use map to apply generate_current_state function to each number in range
    # # the executor will handle creating the threads
    #     args = zip(range(0, longes_run), itertools.repeat(grouped_df), itertools.repeat(columns))
    #     executor.map(generate_current_state, *args)
    #     future_to_num = {executor.submit(generate_current_state, *args): number for number in range(2, longes_run)}

    #     # use concurrent.futures.wait to wait for all threads to complete
    #     concurrent.futures.wait(future_to_num)


def generate_current_state(current_number, grouped_df, columns):
    """Generate the dataset for the current numbers of events, which where triggered."""
    file_name = fr"C:\Users\PhilippKastenhofer\repositories\pm-road-fines\data\new_data\{current_number}_data.csv"
    data = []
    # cols: ['amount', 'org:resource', 'dismissal', 'concept:name', 'vehicleClass',
    # 'totalPaymentAmount', 'lifecycle:transition', 'time:timestamp',
    # 'article', 'points', 'case:concept:name', 'expense', 'notificationType',
    # 'lastSent', 'paymentAmount', 'matricola']
    for idx, group in grouped_df:
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
            pre_df = current_df.combine_first(pre_df)

        entry = pre_df.loc[key].values
        data.append(entry)

    new_df = pd.DataFrame(data)
    print(file_name)
    new_df.to_csv(file_name)


if __name__ == "__main__":
    filepath = r"C:\Users\PhilippKastenhofer\repositories\pm-road-fines\data\logs\log.csv"
    generate(filepath)