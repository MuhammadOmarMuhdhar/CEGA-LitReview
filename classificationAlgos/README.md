# Classification Algorithms

This document details the classification algorithms implemented in this project, with a focus on their functionality and configuration.

Currently Implemented:

*   Few-Shot Classification


## Few-Shot Text Classification

This module provides a function for performing few-shot classification on texts using provided examples. It utilizes sentence transformers for text embedding and K-Nearest Neighbors for classification.

### Dependencies

- pandas
- numpy
- sentence_transformers
- scikit-learn
- logging
- typing

### Function: classify
----
### Description

The `classify` function performs few-shot classification on a series of texts using provided examples.

### Parameters

- `texts` (pd.Series): Series of texts to classify.
- `examples` (List[Dict[str, str]]): List of dictionaries with 'text' and 'label' keys.
- `n_neighbors` (int, optional): Number of neighbors for KNN. Default is 2.
- `confidence_threshold` (float, optional): Minimum confidence threshold. Default is 0.2.
- `model_name` (str, optional): Name of sentence transformer model. Default is 'sentence-transformers/all-MiniLM-L6-v2'.

### Returns

- pd.DataFrame: DataFrame with predictions and confidence scores.

### Raises

- ValueError: If input parameters are invalid.
- RuntimeError: If model loading or prediction fails.

### Process

1. Validates input parameters.
2. Loads the specified sentence transformer model.
3. Prepares example texts and labels.
4. Creates and normalizes embeddings for examples and input texts.
5. Processes input texts in batches to prevent memory issues.
6. For each batch:
   - Initializes and fits a KNN classifier.
   - Gets predictions and confidence scores.
   - Applies the confidence threshold.
7. Combines results from all batches.

### Usage Example

```

import pandas as pd
from your_module import classify
Prepare input texts

texts = pd.Series(["This is a positive review.", "I didn't like the product."])

examples = [
{"text": "Great product!", "label": "positive"},
{"text": "Terrible experience.", "label": "negative"}
]

results = classify(texts, examples)
print(results)

```


### Logging

The module uses Python's logging module to log information and errors. The log level is set to INFO, and the format includes timestamp, log level, and message.

### Error Handling

The function includes comprehensive error handling:
- Input validation errors raise ValueError.
- Model loading and embedding processing errors raise RuntimeError.
- Unexpected errors are logged and re-raised.

## Notes

- The function uses batching to process large numbers of texts efficiently.
- Confidence scores below the threshold result in None values for both label and confidence.
