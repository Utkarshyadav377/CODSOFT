# 🎬 Movie Genre Classification

Predicts movie genre from plot summaries using TF-IDF + ML classifiers.

## Models
- Naive Bayes
- Logistic Regression ← best accuracy (61.6%)
- Linear SVM ← best macro-F1 (0.36)

## Dataset
~54k training movies, 27 genres (drama, thriller, horror, comedy, etc.)

## Usage
```python
import joblib
from train_and_evaluate import predict_genre

pipe = joblib.load("outputs/best_model.joblib")
print(predict_genre("A detective hunts a killer through rainy streets.", pipe))
```

## Requirements
```
pip install scikit-learn pandas numpy matplotlib seaborn joblib
````