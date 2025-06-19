import prepare_data
import dataset
import torch
import trainer
import smolmo
from torch.utils.data import DataLoader
from config import Config

cfg = Config()

csv_file = "Transformer_Trading/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"

df = prepare_data.load_forex_data(csv_file)
df = prepare_data.add_technical_indicators(df)

data_scaled, scaler, feature_cols = prepare_data.select_and_scale_features(df)


ds = dataset.ForexDataset(
    data = data_scaled, 
    input_length = cfg.input_length, 
    output_length = cfg.output_length,
    target_feature_idx = cfg.target_feature_idx
    )

ds_length = len(ds)
train_size  = int(ds_length * 0.8)
val_size    = int(ds_length * 0.1)
test_size   = ds_length - train_size - val_size

# Perform sequential splitting (without shuffling)
train_dataset   = torch.utils.data.Subset(ds, range(0, train_size))
val_dataset     = torch.utils.data.Subset(ds, range(train_size, train_size + val_size))
test_dataset    = torch.utils.data.Subset(ds, range(train_size + val_size, ds_length))

train_loader    = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader      = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
test_loader     = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

model = smolmo.TimeSeriesTransformer(
        num_features = len(feature_cols), 
        d_model = cfg.d_model, 
        input_length = cfg.input_length, 
        nhead = cfg.nhead, 
        dim_feedforward = cfg.dim_feedforward, 
        num_layers = cfg.num_layers, 
        output_length = cfg.output_length,
        device = cfg.device
)

OptimizerCls = getattr(torch.optim, cfg.optimizer_type)
optimizer    = OptimizerCls(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
loss_fn = cfg.loss_fn

print('starting training')
trainer.ForexTrainer(
    model = model, 
    optimizer = optimizer, 
    loss_fn = loss_fn,
    device = cfg.device
    ).train(train_loader = train_loader, num_epochs = cfg.num_epochs)
