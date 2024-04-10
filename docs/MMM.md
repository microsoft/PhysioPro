# MMM

This document gives an example on how to use the PhysioPro framework for finetuning [Learning Topology-Agnostic EEG Representations with Geometry-Aware Modeling](https://seqml.github.io/MMM/).

We'll release the pretrained checkpoint soon. But without it, you can still train the model from scratch on the SEED dataset with MMM.

## Prepare the data

1. Download the dataset: You can apply the SEED dataset on the official website [here](https://bcmi.sjtu.edu.cn/home/seed/).

2. Modify the `data_path` and `save_path` to the original `ExtractedFeatures` folder where you store your downloaded dataset and the target folder you would like to use to store the DE features.

3. Run `scripts/SEED_DE.py` to obtain the compatible format of SEED DE feature. 

4. *Optional: Download the pretrained checkpoint [here](https://seqml.github.io/MMM/) and set the path to the checkpoint in `docs/configs/mmm_emotion.yml`*

## Finetune with MMM
```bash
# create the output directory
mkdir -p outputs/MMM_SEED/7/

# run the finetuning task
python -m physiopro.entry.mmm_emotion docs/configs/mmm_emotion.yml
# or 
# python -m physiopro.entry.mmm_emotion docs/configs/mmm_emotion_from_ckpt.yml 
# if you would like to load the pretrained encoder. 

# tensorboard
tensorboard --logdir outputs/
```

Then it will run finetuning process on the 7th subject. The results will be saved to `outputs/MMM_SEED/7/` directory. You can run the finetuning process similarly on other subjects by changing `data.subject_index` in the configuration.

## Regarding the DE feature
We are now aware of the possible issue of using DE feature for SEED (See this [issue](https://github.com/microsoft/PhysioPro/issues/16)). In order to keep it consistent with our paper, we provide the checkpoint pretrained on DE features but please use it wisely. 

We're working to figure out the influence to our results, as well as training MMM on the raw EEG signals. We'll keep it updated. 