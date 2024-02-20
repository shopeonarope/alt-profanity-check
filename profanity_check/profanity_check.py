"""Profanity check exposed methods"""
from importlib import resources

import numpy as np
import joblib

vectorizer = joblib.load(
    resources.files("profanity_check") / "data" / "vectorizer.joblib"
)
model = joblib.load(
    resources.files("profanity_check") / "data" / "model.joblib"
)


def _get_profane_prob(prob):
    return prob[1]


def predict(texts):
    """Predict texts array"""
    return model.predict(vectorizer.transform(texts))


def predict_prob(texts):
    """Predict texts array returning probabilities"""
    return np.apply_along_axis(
        _get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts))
    )
