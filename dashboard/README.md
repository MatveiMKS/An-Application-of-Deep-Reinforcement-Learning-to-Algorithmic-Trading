# Site local des runs

Lancer des runs (TDQN ou stratégies classiques) depuis le navigateur, puis comparer leurs résultats.

```
conda activate tdqn
python dashboard/server.py
```

Ouvrir ensuite http://localhost:8000. `Ctrl+C` dans le terminal arrête le site, mais pas les runs déjà lancés.

- **File d'attente** : au plus 2 runs tournent en même temps, les autres attendent (« en attente »). Pour en autoriser plus : `python dashboard/server.py --parallel 3`. Les cœurs du processeur sont partagés entre les runs (`OMP_NUM_THREADS`). Les runs en attente repartent si tu relances le site ; ceux qui tournaient quand le site s'est arrêté continuent seuls, mais le site ne peut plus les arrêter.
- **Répétitions (graines)** : N runs identiques avec les graines `graine, graine+1, …` (`random`, `numpy`, `torch`). Sans graine ni répétition, le run n'a pas de graine fixe, comme dans le code d'origine.
- **Balayage** : un paramètre et ses valeurs séparées par `;` (par exemple bruit `0 ; 0, 0.5 ; 0, 1`). Le site crée un run par valeur et par graine, tous dans le même groupe.
- **Synthèse par groupe** (section « Répétitions et balayages ») : moyenne ± écart-type des indicateurs de test et du meilleur Sharpe de validation, par variante, et capital moyen ± 1 écart-type sur la période de test.
- **Progression** : épisodes faits, temps écoulé et temps restant estimé à partir des épisodes déjà terminés (TDQN uniquement).

- Chaque run appelle `TradingSimulator().simulateNewStrategy(...)` du dépôt dans un processus séparé (`run_worker.py`), sans fenêtres matplotlib (`rendering=False`). Pour TDQN, `plotTraining=True` : le site récupère le Sharpe par épisode, la perte et les récompenses (courbes d'entraînement).
- Ses résultats vont dans `dashboard/runs/<id>/` : `config.json`, `status.json`, `log.txt`, `train.csv`, `test.csv`, `metrics.json`, `curves.json`. Pour supprimer un run, supprime son dossier.
- **Options (étapes 3 à 6)** : le formulaire lit les valeurs par défaut dans ton code (`tradingEnv.positionMode`, `dailyShortCost`, `shortEpsilon`, `fractionalShares`, `dataAugmentation.noiseRange`, `tradingSimulator.validationDate`, frais). Ce que tu choisis ne s'applique qu'au run lancé : le worker modifie ces variables en mémoire, tes fichiers ne changent pas. Une option absente de ton code est masquée.
- **Données** : si un CSV de `Data/` manque pour une période (par exemple 2017-2022 et 2022-2023 avec la validation) ou s'il est vide, le worker le construit à partir d'un CSV déjà téléchargé qui couvre la période, au lieu de redemander à Yahoo.
- Les indicateurs viennent de `tradingPerformance.PerformanceEstimator` : toute modification de ce fichier (par exemple 365 jours) s'applique aux runs suivants.
- La page charge Chart.js depuis cdn.jsdelivr.net : il faut une connexion internet.
