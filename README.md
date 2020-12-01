## Mechanisms of Action (MoA) Prediction

Final code for Kaggle [lish-moa](https://www.kaggle.com/c/lish-moa/) competition.

### Description

- Single NN model with few linear layers (see [/src/model.py](src/model.py)),
- Pre-processing/feature engineering: [QuantileTransform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)

### Additional data

```
!pip install ../input/iterativestratification/
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

(only for logging)
!pip install neptune-client
```