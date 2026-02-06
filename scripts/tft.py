import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# 1. Prepare EKS Data (Requires a numeric 'time_idx' and a categorical 'group_id')
df = pd.read_csv('eks_data.csv', parse_dates=['DateTime'])
df['time_idx'] = ((df['DateTime'] - df['DateTime'].min()).dt.total_seconds() // 3600).astype(int)
df['group'] = "cluster_1" # Required even for single-series

# 2. Define the Dataset
max_prediction_length = 6
max_encoder_length = 24

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="CPU_Usage",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["CPU_Usage", "Disk_Usage"],
    time_varying_known_reals=["time_idx"],
    target_normalizer=GroupNormalizer(groups=["group"])
)

# 3. Initialize and Train
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16, 
    attention_head_size=4,
    dropout=0.1,
    loss=QuantileLoss() # Provides p10, p50, p90 ranges
)