## Mechanisms of Action (MoA) Prediction

Final code for Kaggle [lish-moa](https://www.kaggle.com/c/lish-moa/) competition.

### Pre-processing/feature engineering

- Using `g-` (genes) and `c-` (cells) features,
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
- [QuantileTransform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html),
- [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html),

### Training
- `MultilabelStratifiedKFold` instead of traditional sklearn KFold
- Single NN model with only few linear layers (see [/src/model.py](src/model.py)),
- `SmoothBCEwLogits` loss function for training (`torch.nn.BCEWithLogitsLoss` for validation)

### Prediction
- Prediciton clipping (`np.clip(y_hat, 0.001, 0.999)`)
- Averaging results from best folds (best val_loss checkpoint from each fold)

### Additional data

```
!pip install ../input/iterativestratification/
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

(only for logging)
!pip install neptune-client
```