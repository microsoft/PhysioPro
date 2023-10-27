# Datasets

## 'ts' Dataset

TS dataset (see `physiopro.dataset.ts.py`) is designed for general purpose time series classification/regression, e.g., UEA/UCR classification.

A `.ts` file stores time-series data and its metadata.
The description of the schema can be found [here](https://www.sktime.net/en/latest/api_reference/file_specifications/ts.html)

### Explanation of parameters in `ts` Dataset

| Item                | Type  | Description                                    |
| ------------------- | ----- | ---------------------------------------------- |
| data_prefix         | str   | Path to the dataset (must be a directory).     |
| name                | str   | Name of the dataset.                           |
| task                | str   | `classification` or `regression`.              |
| num_classes         | int   | Number of classes for `ClassificationDataset`. |
| max_seq_len         | int   | Max sequence length  in seconds                |
| dataset_split_ratio | float | Percentage of train dataset.                   |
| preprocessor        | list  | See `preprocessor.md`                          |

## 'df' Dataset

The schema of `df` dataset is described [here](dataset/df_schema.md).

### Explanation of parameters in `df` Dataset

| Item          | Type | Description                                    |
| ------------- | ---- | ---------------------------------------------- |
| data_folder   | str  | Path to the dataset (must be a directory).     |
| meta_path     | str  | Path to the meta fiel                          |
| task          | str  | `classification` or `regression`.              |
| num_classes   | int  | Number of classes for `ClassificationDataset`. |
| max_seq_len   | int  | Max sequence length in seconds                 |
| num_variables | int  | Number of variables                            |
| freq          | int  | frequency                                      |
| dataset       | int  | 'train', 'valid', or 'test'                    |
| preprocessor  | list | See `preprocessor.md`                          |


## Preprocessing

See `preprocessor.md` for more information about data preprocessing.
