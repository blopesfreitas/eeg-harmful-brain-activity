# Harmful Brain Activity Classification (EEG)

Project for Harvard's Neuro 140. Classifies seizures and related harmful EEG patterns as a probabilistic six-class problem, evaluated by KL divergence against expert-vote distributions.

## Approach
- **Image branch**: EfficientNetB0 on EEG spectrograms.
- **Time-series branch**: WaveNet on the raw EEG.
- **Ensemble**: weighted average of the two branches.
- **Evaluation**: 5-fold cross-validation; KL divergence vs. expert-vote distributions; error analysis via conditional-probability matrices.

## Files
- `ensemble.py` — the EfficientNetB0 + WaveNet ensemble (readable directly on GitHub)
- `ensemble_colab.ipynb` — the same code as a runnable Colab notebook
- `final_report.pdf` — full report: method, results, and error analysis (start here)

To run with outputs, open `ensemble_colab.ipynb` in [Google Colab](https://colab.research.google.com/), or view it via [nbviewer](https://nbviewer.org/). GitHub's inline notebook preview sometimes fails to load; that's a GitHub rendering quirk, not a problem with the notebook.

## Note
Built on a public Kaggle baseline (EfficientNetB0 starter by Chris Deotte, adapted to PyTorch). My contribution was the ensemble design (adding a raw-EEG time-series branch), the evaluation, and the analysis.
