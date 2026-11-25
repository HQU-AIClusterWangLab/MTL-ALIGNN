from __future__ import annotations

import numpy as np


def binary_entropy(probability, epsilon: float = 1e-7):
    probability = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    return -(
        probability * np.log(probability)
        + (1 - probability) * np.log(1 - probability)
    )


def aggregate_binary_probabilities(
    member_probabilities,
    epsilon: float = 1e-7,
) -> dict[str, np.ndarray]:
    values = np.asarray(member_probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected shape [members, samples].")
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    predictive_entropy = binary_entropy(mean, epsilon)
    expected_entropy = binary_entropy(values, epsilon).mean(axis=0)
    mutual_information = np.maximum(
        predictive_entropy - expected_entropy,
        0.0,
    )
    variation_ratio = 1.0 - np.maximum(
        (values >= 0.5).mean(axis=0),
        (values < 0.5).mean(axis=0),
    )
    return {
        "mean": mean,
        "std": standard_deviation,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
    }
