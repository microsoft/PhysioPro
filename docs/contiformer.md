# CinC2020

This document gives an example on how to use the PhysioPro framework for [ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling](https://seqml.github.io/contiformer/).

The example is based the solution of the 1st ranked team [Prna](https://ieeexplore.ieee.org/document/9344053).


## Scripts

```bash
# prepare data and features
cd data/

# training data
mkdir -p CinC2020/training_data
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/ # 10G
find physionet.org/files/challenge-2020/1.0.2/training -name "*.hea" -exec mv {} CinC2020/training_data \;
find physionet.org/files/challenge-2020/1.0.2/training -name "*.mat" -exec mv {} CinC2020/training_data \;

# features
wget https://physionet.org/static/published-projects/challenge-2020/1.0.2/sources/Prna.zip
unzip Prna.zip && rm Prna.zip
mv Prna/physionet2020-submission/feats Prna/physionet2020-submission/records_stratified_10_folds_v2.csv Prna/physionet2020-submission/top_feats.npy CinC2020
rm -rf Prna

# weights
wget https://raw.githubusercontent.com/physionetchallenges/evaluation-2020/master/weights.csv -P CinC2020

# run training
cd ..
mkdir -p outputs/cinc2020
python -m physiopro.entry.train docs/configs/cinc2020.yml
```