import numpy as np
import pandas as pd

from pathlib import Path

"""
returns the label vector
a device is 'on' if at any time within the timeframe it is running
"""

def create_label_vector(df, start, end, appliances):
    target_vector = np.zeros(len(appliances))
    for i in range(start, end):
        for idx, x in enumerate(df.columns):
            if x not in appliances:
                continue
            else:
                if df[x][i] > 0:
                    target_vector[appliances.index(x)] = 1
    return target_vector

def create_rolling(df, arr, window_size, step, appliances, start=0, drop_zero=True):
    windows = list()
    targets = list()
    arr_length = len(arr)

    if window_size >= arr_length:
        raise Exception("Window size is larger than given array")

    i = 0
    while start < arr_length - 1:
        end = start + window_size
        current_window = arr[start:end]
        if len(current_window) == window_size:
            if drop_zero and not all(entries == 0 for entries in current_window):
                targets.append(create_label_vector(df, start, end, appliances))
                windows.append(current_window)
        start = start + step
        if i % 10000 == 0:
            print(f"Step {i} with array size {arr_length}")
        i += 1

    return windows, targets

def create_timeseries_dataframe(data, window_size, step, appliances):
    windows, targets = create_rolling(
        df=data, arr=list(data["Power"]), window_size=window_size, step=step, appliances=appliances
    )
    column_names_samples = [f"sample_{i}" for i in list(range(window_size))]
    column_names = column_names_samples + appliances
    merged = []
    for i in range(len(windows)):
        merged.append(list(np.array(windows[i])) + list(np.array(targets[i])))
    df_timeseries = pd.DataFrame(data=merged, columns=column_names)
    return df_timeseries

def write_or_load_windows(windowed_data_path, df, window_size, step, appliances):
    file = Path(windowed_data_path)
    if file.exists():
        print(f"Loading existing window file {windowed_data_path}")
        df_timeseries = pd.read_csv(windowed_data_path, index_col=0)
    else:
        print(
            f"Window file not found, creating a new one for window size {window_size} and step size {step}"
        )
        df_timeseries = create_timeseries_dataframe(df, window_size, step, appliances)
        df_timeseries[appliances] = df_timeseries[appliances].astype(int)
        df_timeseries.to_csv(windowed_data_path)
    return df_timeseries