# Contiformer

This document gives an example on how to use the PhysioPro framework for [ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling](https://seqml.github.io/contiformer/).

## Classification

Here we take the `Heartbeat` classification task from [UEA & UCR Time Series Classification Repository](http://timeseriesclassification.com/dataset.php) as an example.

1. Download the dataset

```bash
cd PhysioPro
mkdir data
wget http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip -P data

unzip data/Multivariate2018_ts.zip -d data/
rm data/Multivariate2018_ts.zip
```

2. Run `Heartbeat` classification task with `ContiFormer`

```bash
# create the output directory
mkdir -p outputs/Multivariate_ts/Heartbeat
# run the train task
python -m physiopro.entry.train docs/configs/contiformer_classification.yml
# tensorboard
tensorboard --logdir outputs/
```

The results will be saved to `outputs/Multivariate2018_ts/Heartbeat` directory.
