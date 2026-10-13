# Model comparison (trained on the 70 train cases; mean absolute % error on unseen cases)

## Test set (15 cases)
| Model | Oil cum | Gas cum | Water cum | Oil rate | Water rate | Final oil cum |
|---|---|---|---|---|---|---|
| **Gaussian Process + PCA (used)** | **0.12** | **0.12** | **0.56** | **0.39** | 0.83 | **0.19** |
| Neural network (MLP) | 0.21 | 0.21 | 0.75 | 0.41 | **0.77** | 0.31 |
| Random forest | 0.43 | 0.43 | 1.49 | 0.76 | 1.39 | 0.67 |
| Gradient boosting | 0.46 | 0.46 | 1.15 | 0.80 | 1.25 | 0.73 |
| Ridge, polynomial deg 3 | 0.99 | 1.00 | 3.46 | 1.66 | 2.97 | 1.50 |
| Ridge, polynomial deg 2 | 1.92 | 1.93 | 7.72 | 3.07 | 6.02 | 2.93 |

## Validation set (15 cases)
| Model | Oil cum | Gas cum | Water cum | Oil rate | Water rate | Final oil cum |
|---|---|---|---|---|---|---|
| **Gaussian Process + PCA (used)** | **0.12** | **0.12** | **0.37** | **0.34** | **0.57** | **0.18** |
| Neural network (MLP) | 0.18 | 0.18 | 0.73 | 0.30 | 0.63 | 0.24 |
| Random forest | 0.52 | 0.52 | 2.21 | 0.79 | 1.71 | 0.76 |
| Gradient boosting | 0.42 | 0.42 | 1.18 | 0.72 | 1.17 | 0.65 |
| Ridge, polynomial deg 3 | 1.74 | 1.74 | 6.96 | 2.72 | 5.20 | 2.59 |
| Ridge, polynomial deg 2 | 1.87 | 1.87 | 7.54 | 2.90 | 5.71 | 2.79 |
