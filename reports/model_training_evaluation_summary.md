# Model Training and Evaluation Summary

## Models Compared

- **LSTM sequence model:** Required deep learning model trained from scratch using PyTorch.
- **KNN classifier baseline:** Traditional ML baseline on flattened sequence features.
- **Random forest baseline:** Traditional ML baseline on flattened sequence features.

## Results

```text
              model  accuracy  macro_f1
lstm_sequence_model  0.833187  0.539696
      random_forest  0.719444  0.498349
     knn_classifier  0.775585  0.466332
```

## Evaluation Artifacts

- `reports/tables/model_comparison.csv`
- `reports/tables/lstm_metrics.json`
- `reports/tables/baseline_metrics.json`
- `reports/figures/model_training_loss.png`
- `reports/figures/model_confusion_matrix.png`
- `reports/figures/model_comparison.png`

## Interpretation

The baseline models estimate how much signal can be captured from fixed customer summaries. The LSTM uses the temporal order of weekly behavior, which better matches the customer journey framing of the project. Final interpretation should consider both predictive performance and usefulness of the model outputs in the Streamlit decision-support workflow.
