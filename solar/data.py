"""Validate measurements, retain complete inverter coverage, split by whole days."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'hour_sin', 'hour_cos']
TARGET = 'AC_POWER'

def features(frame):
    out = frame[['IRRADIATION', 'AMBIENT_TEMPERATURE']].copy()
    hour = frame['DATE_TIME'].dt.hour + frame['DATE_TIME'].dt.minute / 60
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURES]

def prepare(generation, weather):
    g, w = generation.copy(), weather.copy()
    g['DATE_TIME'] = pd.to_datetime(g['DATE_TIME'], format='%d-%m-%Y %H:%M')
    w['DATE_TIME'] = pd.to_datetime(w['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    if g['PLANT_ID'].nunique() != 1 or w['PLANT_ID'].nunique() != 1 or g['PLANT_ID'].iloc[0] != w['PLANT_ID'].iloc[0]:
        raise ValueError('Expected matching single-plant data.')
    if g.duplicated(['DATE_TIME', 'SOURCE_KEY']).any() or w.duplicated('DATE_TIME').any():
        raise ValueError('Duplicate timestamp/sensor keys; resolve before training.')
    inverter_count = g['SOURCE_KEY'].nunique()
    valid_g = np.isfinite(g['AC_POWER']) & g['AC_POWER'].ge(0)
    grouped = g[valid_g].groupby('DATE_TIME').agg(AC_POWER=('AC_POWER', 'sum'), count=('SOURCE_KEY', 'nunique'))
    complete = grouped[grouped['count'].eq(inverter_count)].drop(columns='count')
    joined = complete.reset_index().merge(w[['DATE_TIME', 'IRRADIATION', 'AMBIENT_TEMPERATURE']], on='DATE_TIME', validate='one_to_one')
    valid = np.isfinite(joined[['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE']]).all(axis=1) & joined['IRRADIATION'].ge(0)
    cleaned = joined[valid].sort_values('DATE_TIME').reset_index(drop=True)
    audit = {'raw_generation_rows': len(g), 'inverters': inverter_count, 'generation_timestamps': g['DATE_TIME'].nunique(),
             'complete_timestamps': len(complete), 'matched_timestamps': len(joined), 'retained_timestamps': len(cleaned),
             'invalid_generation_rows': int((~valid_g).sum())}
    return cleaned, audit

def load(directory='data/raw'):
    p = Path(directory)
    return prepare(pd.read_csv(p / 'Plant_1_Generation_Data.csv'), pd.read_csv(p / 'Plant_1_Weather_Sensor_Data.csv'))

def split_days(frame):
    days = frame['DATE_TIME'].dt.normalize()
    unique = np.sort(days.unique())
    if len(unique) < 10:
        raise ValueError('At least 10 distinct days required.')
    a, b = int(len(unique) * .6), int(len(unique) * .8)
    return (frame[days < unique[a]].copy(), frame[(days >= unique[a]) & (days < unique[b])].copy(), frame[days >= unique[b]].copy())
