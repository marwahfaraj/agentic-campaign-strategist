# 4 Methodology (draft for the final report)

*Draft for ticket #10. Written in report prose per the capstone template. Numbers that belong in
the Results section are intentionally left out. Citations marked (CITE) need the full reference
added to Works Cited.*

---

This project frames audience-to-campaign fit as a sequence classification problem: given twelve
weeks of a household's purchase behavior, the model predicts which of six campaign activation
styles the household is most likely to respond to in the following four weeks. Predictions for
individual households are then aggregated to the audience level to produce a single recommendation
and confidence estimate for the marketer.

The cleaned Complete Journey transactions (84.51°, n.d.) were converted into weekly journey
sequences. Each household-week was summarized by nineteen features: an activity indicator;
log-scaled spend, quantity, and trip counts; a discount rate; a coupon-use indicator; two cyclical
calendar features encoding week-of-year; a recency counter (weeks since last activity); an active
streak counter; and the household's spend share across the eight highest-revenue product categories
plus an aggregate other category. Log scaling was applied to spend-related features to reduce the
strong right skew observed during exploratory analysis, and the calendar, recency, and streak
features are computed only from information available at prediction time. A sliding window of
twelve consecutive weeks forms one model input, and windows with fewer than two active weeks were
excluded because such households show too little behavior to be part of a targeted marketing
audience.

Because the public dataset contains no campaign response outcomes, activation-style labels were
generated with weak supervision (Ratner et al., 2017). Each window's label is decided by rules that
read only the four weeks after the window: for example, a household that was regularly active
during the window but silent in all four following weeks is labeled a win-back case, while a
household whose future purchases show a strong discount share is labeled price-led. Windows with no
clear future signal are abstained from and excluded from training, which is standard
weak-supervision practice. This future-window design is a deliberate leakage control: an earlier
prototype computed labels from signals inside the input window, which allowed models to re-learn
the labeling rules and produced misleadingly high scores. Under the final design, a model can only
perform well by finding past behavior patterns that genuinely predict future, marketing-relevant
behavior. The rule thresholds were reviewed against the real data during development; in one case,
a seasonal window inherited from an earlier synthetic simulator was removed after the data showed
no sales lift in those weeks.

The deep learning model is a bidirectional long short-term memory network (Hochreiter &
Schmidhuber, 1997) implemented from scratch in PyTorch (CITE Paszke et al., 2019). Two stacked
bidirectional LSTM layers encode the weekly sequence, an additive attention layer (CITE Bahdanau et
al., 2015) pools the weekly hidden states into a single representation, and a two-layer classifier
head with layer normalization, ReLU activation, and dropout maps that representation to the six
activation styles. The attention pooling was chosen over using only the final hidden state so that
the model can weight informative weeks (for example, the week a household went silent) regardless
of where they occur in the window. Model capacity (hidden size) was treated as a hyperparameter
rather than fixed in advance.

Four reference models provide context for the deep learning results, all implemented with
scikit-learn (CITE Pedregosa et al., 2011): a majority-class predictor and a stratified random
predictor establish the naive floor, and K-nearest neighbors and random forest classifiers form
strong tabular baselines. The tabular models receive flattened sequence summaries (per-feature
mean across weeks, final-week values, and start-to-end trend), so the comparison isolates the value
of week-order information that only the sequence model can use.

Data were split by household into training (65%), validation (15%), and test (20%) sets using
grouped splitting, so that overlapping windows from the same household can never appear on both
sides of a split; the split code asserts this property at run time. The feature scaler was fit on
the training split only and then applied to the validation and test splits. Training used the AdamW
optimizer with weight decay, a cosine annealing learning-rate schedule, gradient clipping, a batch
size of 64, and a cross-entropy loss with mild label smoothing and square-root inverse-frequency
class weights to address the class imbalance identified during labeling. Models trained for up to
25 epochs with early stopping on validation macro-F1 (patience of five epochs), restoring the best
checkpoint. Macro-F1 was chosen as the selection and headline metric because the activation-style
classes are imbalanced and every class matters equally to the marketing use case.

Hyperparameter optimization searched a grid over hidden size (64, 96, 128) and learning rate
(0.0005, 0.0008, 0.001), with each configuration trained to convergence under the same early
stopping rule and compared on validation macro-F1. The best configuration was then retrained on the
combined training and validation data, holding out a small internal slice for early stopping, and
evaluated exactly once on the held-out test set.

Finally, two verification steps confirm that the methodology was applied correctly. First, the
naive baselines bound the problem: the majority-class predictor achieves moderate accuracy but
near-zero macro-F1, confirming that accuracy alone would be a misleading metric. Second, a
shuffled-label control repeats the full LSTM training procedure on randomly permuted training
labels; its test macro-F1 collapses to chance level, demonstrating that no information leaks from
features, splits, or preprocessing, and that the reported performance reflects genuine
label-behavior signal. Both checks are reproducible in the project notebooks.

---

*Works Cited entries needed: 84.51° (dataset); Ratner et al. 2017 (Snorkel/weak supervision);
Hochreiter & Schmidhuber 1997 (LSTM); Bahdanau et al. 2015 (attention); Paszke et al. 2019
(PyTorch); Pedregosa et al. 2011 (scikit-learn).*
