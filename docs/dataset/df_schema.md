# Schema of `df` Dataset

## File Structure

```
dataset_name
│
├───feature_line
│   ├───{id}.pkl
│
├───label_line
│   ├───{id}.pkl
│
├───meta.pkl
```

One file for each sample, features and labels are placed in different folders.

## File Schema

The first column of a feature/label file is the identifier.

Example of feature file, shape = (feature_num \* sequence_length \* frequency, sample_num):

```csv
        F1      F2      F3
abcedf  19.836410 -18.005357
abcedf  32.653783  10.070793
abcedf  13.523455  32.653783
```

Example of label file, shape = (1, sample_num):

```csv
    label
abcedf  0
abcedf  1
abcedf  0
```

Example of meta:

```
{
    'classes': [0, 1],
    'train': ['abcedf', 'bdadfe', ...], # identifiers of train dataset
    'test': ['cxzvcv', ...], # identifiers of test dataset
    'mean@train': F1    12.234567
                F2  32.653783
                F3  23.456789
    'std@train': F1 5.234567
                F2  8.456789
                F3  12.345678
}
```
