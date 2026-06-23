# Model Optimization Summary

The optimization experiment compares a small set of LSTM configurations. The search is intentionally limited because the capstone timeline is seven weeks and the project also includes an end-to-end application layer.

## Best Configuration

```json
{
  "run": 4,
  "data_source": "synthetic retail simulator",
  "sequence_length": 16,
  "hidden_size": 48,
  "learning_rate": 0.0007,
  "epochs": 1,
  "accuracy": 0.8200980392156862,
  "macro_f1": 0.5257305829331523,
  "training_samples": 30600
}
```

## Output Artifacts

- `reports/tables/optimization_results.csv`
- `reports/tables/best_hyperparameters.json`
- `reports/figures/optimization_macro_f1.png`

## Interpretation

Macro-F1 is used as the primary optimization metric because the activation-style labels can be imbalanced. The best configuration can be used for the final training run and discussed in the experimental methods section.
