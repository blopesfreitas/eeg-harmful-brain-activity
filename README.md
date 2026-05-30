# Harmful Brain Activity Classification (EEG)

Project for Harvard's Neuro 140. Classifies seizures and related harmful EEG
patterns as a probabilistic six-class problem, evaluated by KL divergence
against expert-vote distributions.

## Approach
- **Image branch**: EfficientNetB0 on EEG spectrograms.
- **Time-series branch**: WaveNet on the raw EEG.
- **Ensemble**: weighted average of the two branches.
- **Evaluation**: 5-fold cross-validation; KL divergence vs. expert-vote
  distributions; error analysis via conditional-probability matrices.

## Files
- `ensemble.ipynb` — the EfficientNetB0 + WaveNet ensemble
- `final_report.pdf` — full write-up: method, results, and error analysis

## Note
Built on a public Kaggle baseline (EfficientNetB0 starter by Chris Deotte,
adapted to PyTorch). My contribution was the ensemble design (adding a raw-EEG
time-series branch), the evaluation, and the analysis.
