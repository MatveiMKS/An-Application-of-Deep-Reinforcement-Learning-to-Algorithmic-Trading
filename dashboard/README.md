# Site local des runs

Lancer des runs (TDQN ou stratégies classiques) depuis le navigateur, puis comparer leurs résultats.

```
conda activate tdqn
python dashboard/server.py
```

Ouvrir ensuite http://localhost:8000. `Ctrl+C` dans le terminal arrête le site, mais pas les runs déjà lancés.

- Chaque run appelle `TradingSimulator().simulateNewStrategy(...)` du dépôt dans un processus séparé (`run_worker.py`), sans graphiques matplotlib (`rendering=False`, `plotTraining=False`).
- Ses résultats vont dans `dashboard/runs/<id>/` : `config.json`, `log.txt`, `train.csv`, `test.csv`, `metrics.json`. Pour supprimer un run, supprime son dossier.
- Les indicateurs viennent de `tradingPerformance.PerformanceEstimator` : toute modification de ce fichier (par exemple 365 jours) s'applique aux runs suivants.
- La page charge Chart.js depuis cdn.jsdelivr.net : il faut une connexion internet.
