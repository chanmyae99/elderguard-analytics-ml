"""
feature_engineer.py
-------------------
Creates additional derived features from the raw sensor readings
to improve model discriminability.

Engineered features
-------------------
CO2_diff      : Difference between the two CO2 sensor readings.
                The two sensors use different technologies (infrared vs
                electrochemical), so their gap reflects measurement drift
                and may encode environmental context.

MOS_mean      : Mean of the four Metal Oxide Sensor units.
                Captures the overall VOC / gas load in the room.

MOS_std       : Standard deviation across the four MOS units.
                High variance may indicate a localised gas source
                (e.g. cooking), which correlates with higher activity.

MOS_max       : Maximum MOS reading across units.
                Highlights peak gas events more sensitively than the mean.

Temp_x_CO2    : Interaction between temperature and CO2.
                Body heat and breathing both rise with physical activity,
                making their product a stronger activity signal than either
                feature alone.

CO2_x_MOS     : Interaction between CO2 and MOS_mean.
                Combined respiratory and VOC output is a proxy for
                the intensity of occupant activity.

Author: [Your Name]
"""

import pandas as pd

MOS_COLS = [
    "MetalOxideSensor_Unit1",
    "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append engineered features to the DataFrame in-place and return it.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, imputed dataset (numeric columns already filled).
        Must contain the raw sensor columns listed in MOS_COLS plus
        Temperature, CO2_InfraredSensor, and CO2_ElectroChemicalSensor.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new feature columns appended.
    """
    df = df.copy()

    df["CO2_diff"]   = df["CO2_ElectroChemicalSensor"] - df["CO2_InfraredSensor"]
    df["MOS_mean"]   = df[MOS_COLS].mean(axis=1)
    df["MOS_std"]    = df[MOS_COLS].std(axis=1)
    df["MOS_max"]    = df[MOS_COLS].max(axis=1)
    df["Temp_x_CO2"] = df["Temperature"] * df["CO2_ElectroChemicalSensor"]
    df["CO2_x_MOS"]  = df["CO2_ElectroChemicalSensor"] * df["MOS_mean"]

    print(f"[feature_engineer] Added 6 engineered features. "
          f"Total features: {df.shape[1]}")
    return df
