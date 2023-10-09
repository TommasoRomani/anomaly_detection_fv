import numpy as np
import pandas as pd
import sranodec as anom


def clean_data(data:pd.DataFrame):
    # less than period
    amp_window_size=24
    # (maybe) as same as period
    series_window_size=24
    # a number enough larger than period
    score_window_size=100
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
    score = spec.generate_anomaly_score(data)
    return data

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
