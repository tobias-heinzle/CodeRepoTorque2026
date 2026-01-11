
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

def load_bearing_fault_samples(wind_farm: str = 'B', anomaly_string: str = 'bearing') -> list[pd.DataFrame]:
    base_dir = Path(f"dataset/")
    data_dir = base_dir / f"Wind Farm {wind_farm}"
    event_info = pd.read_csv(data_dir / "event_info.csv", sep=";")

    samples = []

    relevant_events = list(
        event_info.iloc()
        )

    for event in tqdm(relevant_events):
        scada = pd.read_csv(data_dir / f"datasets/{event["event_id"]}.csv", sep=";")
        scada["time_stamp"] = pd.to_datetime(scada["time_stamp"])

        # We only consider the anomalies regarding bearing failures, other data is considered normal w.r.t. main bearing behavior
        scada["label"] = 'anomaly' if (anomaly_string.lower() in str(event["event_description"]).lower()) else 'normal'
        scada["asset"] = event["asset_id"]
        scada["event_description"] = event["event_description"]
        samples.append(scada.copy())

    return samples


def load_daily_indicator_data(data_dir: Path = Path("indicator_data/lin_reg")) -> list[pd.DataFrame]:
    
    data = []
    threshold = .9

    for file in os.listdir(data_dir):
        df = pd.read_csv(data_dir / file)
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        cutoff = df.training_cutoff.iloc[0]

        # If the R2 score on the training segement is below 0.9, 
        # we assume the NBM is not performing well and it is cut
        score = r2_score(df.target.to_numpy()[:cutoff], df.predictions.to_numpy()[:cutoff])

        if score < threshold:
            print(file, 'excluded from training, r2 =', round(score, 3), '<', threshold)
            continue

        data += [df]

    daily_time_series = []
    batch_size = 6 * 24
    max_len = 324

    for df in data:
        ys = df["residual"].to_numpy()
        ys_batched = np.convolve(ys, np.ones(batch_size)/batch_size, mode='valid')[::batch_size]
        if df['label'].iloc[0] != 'normal':
            ys_batched = np.concat([ys_batched, [1.0, 1.0]])
        ys_batched = ys_batched[-max_len:]
        daily_time_series.append(ys_batched)

    return np.array(daily_time_series)