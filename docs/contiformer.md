# Contiformer

This document gives an example on how to use the PhysioPro framework for [ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling](https://seqml.github.io/contiformer/).

## Run Irregular Time Series Classification

Here we take the `Heartbeat` classification task from [UEA & UCR Time Series Classification Repository](http://timeseriesclassification.com/dataset.php) as an example.

1. Download the dataset

```bash
cd PhysioPro
mkdir data
wget http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip -P data

unzip data/Multivariate2018_ts.zip -d data/
rm data/Multivariate2018_ts.zip
```

2. Run irregular time series classification task with `ContiFormer`

```bash
# create the output directory
mkdir -p outputs/Multivariate_ts/Heartbeat
# run the train task
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0.3 --data.name Heartbeat
# tensorboard
tensorboard --logdir outputs/
```

The results will be saved to `outputs/Multivariate_ts/Heartbeat` directory.

3. Run regular time series classification task with `ContiFormer`

```bash
# create the output directory
mkdir -p outputs/Multivariate_ts/Heartbeat
# run the train task
python -m physiopro.entry.train docs/configs/contiformer_classification.yml --data.name Heartbeat
# or use the following command
python -m physiopro.entry.train docs/configs/contiformer_mask_classification.yml --data.mask_ratio 0 --data.name Heartbeat
# tensorboard
tensorboard --logdir outputs/
```

The results will be saved to `outputs/Multivariate_ts/Heartbeat` directory.
