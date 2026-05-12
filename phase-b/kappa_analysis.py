import pandas as pd
from sklearn.metrics import cohen_kappa_score

pairs = pd.read_csv('phase-b/pairwise_results.csv').head(10)
human = pd.read_csv('phase-b/human_labels.csv')
kappa = cohen_kappa_score(human['human_winner'], pairs['winner_after_swap'])
print(f"Cohen's kappa: {kappa:.3f}")
print('Interpretation: substantial agreement; production monitoring candidate.' if kappa >= 0.6 else 'Interpretation: needs more calibration data.')
