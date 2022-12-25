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

def fft_aggregated(x, param):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    assert {config["aggtype"] for config in param} <= {
        "centroid",
        "variance",
        "skew",
        "kurtosis",
    }, 'Attribute must be "centroid", "variance", "skew", "kurtosis"'

    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y), dtype=float) ** moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 3) - 3 * get_centroid(y) * variance - get_centroid(y) ** 3
            ) / get_variance(y) ** (1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 4)
                - 4 * get_centroid(y) * get_moment(y, 3)
                + 6 * get_moment(y, 2) * get_centroid(y) ** 2
                - 3 * get_centroid(y)
            ) / get_variance(y) ** 2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis,
    )

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in param]
    index = ["freq_{}".format(config["aggtype"]) for config in param]
    return zip(index, res)

def get_mean_change(x):
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN

def count_non_zero_rows(dataframe, column):
    return len(dataframe) - dataframe[column].isin([0]).sum()

