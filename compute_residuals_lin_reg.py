import os
from pathlib import Path

import pandas as pd
import numpy as np

import pandas as pd
from tqdm import tqdm

from load_data import load_bearing_fault_samples

def design_matrix(rot_speed, env_temp, ref_temp, power = None):
    
    x = rot_speed
    y = env_temp
    z = ref_temp

    if power is not None:
        u = power
        return np.stack([np.ones_like(x), x, y, z, u, u*x, u*y, u*z, z*x, z*y, x*y])
    else:
        return np.stack([np.ones_like(x), x, y, z, z*z, z*x, z*y, z*x*y, x*x, y*y, x*y, x*x*y, y*y*x, x*x*y*y])

def normalize(data, reference=None):
    if reference is None:
        reference = data.copy()
    data = np.array(data)
    reference = np.array(reference)

    lower = np.quantile(reference, 0.01)
    upper = np.quantile(reference, 0.99)

    data = np.clip(data, lower, upper)
    data = (data - lower) / (upper - lower)
    return data

def compute_residuals(sample: pd.DataFrame) -> pd.DataFrame:

    valid_status_ids = [0, 2]
    training_days = 300
    train_cutoff = 6 * 24 * training_days


    target_label = 'sensor_52_avg' # Rotor bearing 2 temperature
    env_temp_label = "sensor_8_avg" # Outside temperature
    ref_temp_label = 'sensor_51_avg' # Rotor bearing 1 temperature
    power_label = "power_62_avg" # Active power
    rot_speed_label = "sensor_25_avg" # Rotor speed average 
    time_label = 'time_stamp'

    full_df = sample.fillna(0.0)
    description = full_df['event_description'].iloc[0]
    label = full_df["label"].iloc[0]
    asset = full_df["asset"].iloc[0]

    full_df = full_df[full_df['status_type_id'].isin(valid_status_ids)]
    
    train_df = full_df[full_df["train_test"] == "train"]
    test_df = full_df[full_df["train_test"] == "prediction"]

    test_period_start = len(train_df)

    for l in (target_label, env_temp_label, ref_temp_label, power_label, rot_speed_label):
        test_df.loc[:, l] = normalize(test_df[l], reference=train_df[l])
        train_df.loc[:, l] = normalize(train_df[l])

    time = train_df[time_label]
    rot_speed = train_df[rot_speed_label]
    env_temp = train_df[env_temp_label]
    ref_temp = train_df[ref_temp_label]
    target_temp = train_df[target_label]
    power = train_df[power_label]

    time_test = test_df[time_label]
    rot_speed_test = test_df[rot_speed_label]
    env_temp_test = test_df[env_temp_label]
    ref_temp_test = test_df[ref_temp_label]
    target_temp_test = test_df[target_label]
    power_test = test_df[power_label]

    time_full = np.concatenate([time, time_test])
    rot_speed_full = np.concatenate([rot_speed, rot_speed_test])
    env_temp_full = np.concatenate([env_temp, env_temp_test])
    ref_temp_full = np.concatenate([ref_temp, ref_temp_test])
    target_temp_full= np.concatenate([target_temp, target_temp_test])
    power_full = np.concatenate([power, power_test])

    # Now training a simple linear regression model, only on the train portion

    X = design_matrix(rot_speed[:test_period_start], env_temp[:test_period_start], ref_temp[:test_period_start], power[:test_period_start])
    X_full = design_matrix(rot_speed_full, env_temp_full, ref_temp_full, power_full)

    w = np.linalg.solve(X @ X.T, X @ target_temp[:test_period_start]) 

    z_pred = w @ X_full



    return  pd.DataFrame.from_dict({
        'time_stamp' : pd.to_datetime(time_full),
        'residual' : np.abs(target_temp_full - z_pred),
        'target' : target_temp_full,
        'predictions' : z_pred,
        'description': description,
        'label': label,
        'training_cutoff' : train_cutoff,
        'test_period_start': test_period_start,
        'asset' : asset
    })


def save_residuals():
    experiment_name = 'lin_reg'
    
    samples = load_bearing_fault_samples(wind_farm='B')
    base_dir = Path('indicator_data')

    os.makedirs(base_dir / experiment_name, exist_ok=True)

    for k, sample in enumerate(tqdm(samples)):
        res_data = compute_residuals(sample)
        res_data.to_csv(base_dir / experiment_name / f'residuals_{k}.csv') 



if __name__ == "__main__":
    save_residuals()