## Preprocessor

Defines a combination of transformation or normalization methods for pre-precessing time series data.

### A demo of configuring a preprocessor using yaml

```
data:
  preprocessor:
    online_preprocess: true
    x_transformation:
      - type: zscore
        axis: -1
      - type: downsample
        axis: -2
        ratio: 4
      - type: upsample
        axis: -2
        ratio: 2
      - type: downsample
        axis: -3
        ratio: 2
    sample_ratio: 0.1
```

Also see `./tests/transformer_forecaster.yml`

### Explanation of schema in preprocessor

| Item              | Type      | Description                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| online_preprocess | bool      | If `true`, the operation will be executed during `__getitem__`, otherwise it will be executed during the loading of the dataset. For certain transformations (such as FFT) that require significant additional space and time overhead, online preprocess is favoured, the default value is `true`.                                                                                      |
| x_transformation  | list/None | Define a series of operations to be performed on features of a dataset, the default value is `None`.                                                                                                                                                                                                                                                                                     |
| y_transformation  | list/None | Define a series of operations to be performed on labels of a dataset, the default value is `None`.                                                                                                                                                                                                                                                                                       |
| sample_ratio      | float     | Perform data sampling for training dataset in order to change the ratio of positive and negative samples. If `sample_ratio>1`, perform over-sampling for positive samples, otherwise, perform down-sampling for negative samples, the default value is `1`. Note that it is only available for binary classification task with imbalance data and #negative samples > #positive samples. |

### Schema of implemented transformations

#### z-score: Z-Score Normalization

* type: zscore
* axis: the axis to preform normalization, default to `-1` (last dimension).

#### minimax: MiniMax Normalization

* type: minimax
* axis: the axis to preform normalization, default to `-1` (last dimension).

#### fft: Fourier Transformation

* type: fft
* axis: the axis to preform normalization, default to `-2`.

#### downsample: Downsampling Transformation

* type: downsample
* axis: the axis to preform normalization, default to `-2`.
* ratio: the ratio to preform downsampling, default to `1`.

#### upsample: Upsampling Transformation

* type: upsample
* axis: the axis to preform normalization, default to `-2`.
* ratio: the ratio to preform upsampling, default to `1`.

#### Add your own transformation

See `physiopro/dataset/transformation` to register your transformation.
