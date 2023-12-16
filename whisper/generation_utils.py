import numpy as np


def softmax(logits, axis=-1) -> np.ndarray :
    """
    Compute the softmax function.

    Args:
    - logits: Input logits.
    - axis: Axis along which the softmax operation is performed.

    Returns:
    - Softmax probabilities.
    """
    logits_max = np.max(logits, axis=axis, keepdims=True)
    logits = logits - logits_max
    logits_exp = np.exp(logits)
    prob = logits_exp / logits_exp.sum(axis=axis, keepdims=True)
    return prob


def log_softmax(logits, axis=-1) -> np.ndarray:
    """
    Compute the log softmax function.

    Args:
    - logits: Input logits.
    - axis: Axis along which the log softmax operation is performed.

    Returns:
    - Log softmax probabilities.
    """
    logits_max = np.max(logits, axis=axis, keepdims=True)
    logits = logits - logits_max
    return logits - np.log(np.exp(logits).sum(axis=axis, keepdims=True))


def sample_top_p(probs, p=0.95, size=1) -> np.ndarray:
    """
    Sample from the top-p distribution.

    Args:
    - probs: Probabilities.
    - p: Threshold probability for selecting tokens.
    - size: Number of samples to generate.

    Returns:
    - Sampled indices.
    """
    probs = probs.reshape(-1)
    # Sort the logits in descending order
    sorted_indices = (np.argsort(probs)[::-1]).astype(np.int32)  
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)
    
    # Select indices based on the top-p sampling
    sorted_probs = sorted_probs[cumulative_probs < p]
    sorted_indices = sorted_indices[cumulative_probs < p]
    sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
    indices = np.random.choice(sorted_indices, size, p=sorted_probs)
    
    if size == 1:
        return indices[0]
    return indices
