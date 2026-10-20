# coding=utf-8

"""
Lance UN run (entraînement + test) avec le TradingSimulator du dépôt et
enregistre ses résultats dans dashboard/runs/<id>/.

Appelé par server.py dans un processus séparé :
    python run_worker.py <dossier_du_run>

Le dossier contient config.json (écrit par le serveur). Le worker y écrit :
    status.json   état du run (running / done / error)
    train.csv     activité de trading sur la période d'entraînement
    test.csv      activité de trading sur la période de test
    metrics.json  indicateurs de PerformanceEstimator (train et test)
    curves.json   courbes d'entraînement de TDQN, mises à jour pendant le run
Les impressions du code (tqdm, tableaux) partent dans log.txt via le serveur.
"""

import json
import os
import sys
import time
import traceback

RUN_DIR = os.path.abspath(sys.argv[1])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Le code du dépôt utilise des chemins relatifs ('Data/', 'Figures/', 'runs/')
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')


def writeStatus(state, **extra):
    status = {'state': state, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    status.update(extra)
    with open(os.path.join(RUN_DIR, 'status.json'), 'w') as f:
        json.dump(status, f)


def metrics(data):
    """Indicateurs numériques de tradingPerformance.PerformanceEstimator."""
    from tradingPerformance import PerformanceEstimator
    analyser = PerformanceEstimator(data)
    analyser.computePerformance()
    values = {
        'PnL': analyser.PnL,
        'Annualized Return (%)': analyser.annualizedReturn,
        'Annualized Volatility (%)': analyser.annualizedVolatily,
        'Sharpe Ratio': analyser.sharpeRatio,
        'Sortino Ratio': analyser.sortinoRatio,
        'Max Drawdown (%)': analyser.maxDD,
        'Max Drawdown Duration (days)': analyser.maxDDD,
        'Profitability (%)': analyser.profitability,
        'Avg Profit/Loss Ratio': analyser.averageProfitLossRatio,
        'Skewness': analyser.skewness,
    }
    # JSON n'accepte ni NaN ni l'infini
    return {k: (float(v) if v == v and abs(float(v)) != float('inf') else None) for k, v in values.items()}


class Curves:
    """
    Courbes d'entraînement de TDQN, récupérées sans modifier TDQN.py :
    - Sharpe par épisode : TDQN les envoie à tensorboard avec
      self.writer.add_scalar(...) (TDQN.py:736 et 743 dans l'original). On
      remplace SummaryWriter par Recorder, qui les écrit dans curves.json.
      Cela évite aussi le nom de dossier de l'original
      ('runs/' + '%d/%m/%Y-%H:%M:%S', TDQN.py:380), qui contient des « : »
      interdits dans un chemin Windows.
    - perte (Huber) de chaque mise à jour du réseau : on intercepte
      F.smooth_l1_loss (TDQN.py:603) et on enregistre sa moyenne par paquets.
    - récompense totale par épisode : TDQN la passe à self.plotTraining(score, ...)
      à la fin de l'entraînement (TDQN.py:769).
    """

    lossChunk = 100   # nombre de mises à jour moyennées par point de la courbe de perte

    def __init__(self):
        self.data = {'scalars': {}, 'loss': [], 'rewards': [], 'lossChunk': self.lossChunk}
        self.losses = []
        self.updates = 0

    def save(self):
        path = os.path.join(RUN_DIR, 'curves.json')
        with open(path + '.tmp', 'w') as f:
            json.dump(self.data, f)
        os.replace(path + '.tmp', path)

    def addScalar(self, tag, value, step):
        self.data['scalars'].setdefault(tag, []).append([int(step), float(value)])
        self.save()

    def addLoss(self, value):
        self.losses.append(value)
        self.updates += 1
        if len(self.losses) == self.lossChunk:
            self.data['loss'].append([self.updates, sum(self.losses) / len(self.losses)])
            self.losses = []
            if len(self.data['loss']) % 10 == 0:
                self.save()

    def addRewards(self, score):
        self.data['rewards'].append([float(x) for x in score])
        self.save()


curves = Curves()


class Recorder:
    """Remplace torch.utils.tensorboard.SummaryWriter dans TDQN.py."""
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        curves.addScalar(tag, value, step if step is not None else 0)
    def close(self):
        curves.save()
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FunctionalProxy:
    """Remplace torch.nn.functional dans TDQN.py pour lire la perte."""
    def __init__(self, functional):
        self.functional = functional
    def smooth_l1_loss(self, *args, **kwargs):
        loss = self.functional.smooth_l1_loss(*args, **kwargs)
        curves.addLoss(loss.item())
        return loss
    def __getattr__(self, name):
        return getattr(self.functional, name)


def instrumentTDQN():
    import TDQN as tdqnModule
    tdqnModule.SummaryWriter = Recorder
    tdqnModule.F = FunctionalProxy(tdqnModule.F)
    originalPlot = tdqnModule.TDQN.plotTraining
    def plotTraining(self, score, marketSymbol):
        curves.addRewards(score)
        return originalPlot(self, score, marketSymbol)
    tdqnModule.TDQN.plotTraining = plotTraining


def applyOptions(config):
    """
    Options ajoutées aux étapes 3 à 6 du guide : ce sont des variables globales
    des modules du dépôt, qu'on modifie ici pour ce run seulement (le worker
    est un processus à part). Une option absente du code est ignorée, pour que
    le site marche aussi sur une version plus ancienne du dépôt.
    Renvoie les valeurs réellement utilisées.
    """
    import tradingEnv, dataAugmentation, tradingSimulator
    used = {}

    def setOption(module, name, value):
        if value is None or not hasattr(module, name):
            return
        setattr(module, name, value)
        used[name] = value

    # tradingEnv.py (étape 4)
    mode = config.get('positionMode')
    if mode and mode in getattr(tradingEnv, 'positionModes', ()):
        setOption(tradingEnv, 'positionMode', mode)
    if config.get('dailyShortCost') not in (None, ''):
        setOption(tradingEnv, 'dailyShortCost', float(config['dailyShortCost']))
    epsilon = str(config.get('shortEpsilon') or '').strip()
    if epsilon:
        setOption(tradingEnv, 'shortEpsilon', 'auto' if epsilon == 'auto' else float(epsilon))
    if config.get('fractionalShares') is not None:
        setOption(tradingEnv, 'fractionalShares', bool(config['fractionalShares']))

    # dataAugmentation.py (étape 3) : liste de écarts-types en % du prix
    noise = str(config.get('noiseRange') or '').strip()
    if noise:
        setOption(dataAugmentation, 'noiseRange', [float(x) for x in noise.replace(';', ',').split(',') if x.strip()])

    # tradingSimulator.py (étape 6) : lue au moment de l'appel à simulateNewStrategy
    if config.get('validationDate'):
        setOption(tradingSimulator, 'validationDate', config['validationDate'])

    return used


def marketSymbol(stockName):
    """Symbole Yahoo d'un actif du simulateur (None pour les actifs fictifs)."""
    import tradingSimulator as ts
    for name in ('cryptos', 'indices', 'companies', 'stocks'):
        table = getattr(ts, name, {})
        if stockName in table:
            return table[stockName]
    return None


def csvRange(path):
    """(première date, dernière date) d'un CSV de Data/, ou None s'il est vide ou illisible."""
    import pandas as pd
    try:
        data = pd.read_csv(path, header=0, index_col='Timestamp', parse_dates=True)
    except Exception:
        return None
    if data.empty:
        return None
    return data.index[0], data.index[-1]


def prepareData(symbol, ranges):
    """
    Le code du dépôt cherche un fichier Data/<symbole>_<début>_<fin>.csv par
    période et le télécharge sur Yahoo s'il n'existe pas. Avec la validation
    (étape 6), il faut de nouvelles périodes (2017-2022 et 2022-2023), et
    yfinance 0.2.57 se fait souvent refuser par Yahoo (faux « rate limit »),
    ce qui laisse un CSV vide. On construit donc ces fichiers à partir des CSV
    déjà téléchargés qui couvrent la période. Un CSV vide est remplacé de la
    même façon. Sans source qui couvre la période, on laisse le code télécharger.
    """
    import glob
    import pandas as pd
    for start, end in ranges:
        target = os.path.join('Data', '{}_{}_{}.csv'.format(symbol, start, end))
        if os.path.isfile(target) and csvRange(target) is not None:
            continue
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        for source in glob.glob(os.path.join('Data', symbol + '_*.csv')):
            span = csvRange(source)
            if source == target or span is None:
                continue
            # Yahoo exclut la date de fin : la source doit aller jusqu'à la veille
            if span[0] <= first + pd.Timedelta(days=4) and span[1] >= last - pd.Timedelta(days=4):
                data = pd.read_csv(source, header=0, index_col='Timestamp', parse_dates=True)
                data[(data.index >= first) & (data.index < last)].to_csv(target)
                print('Données : {} construit à partir de {}'.format(target, source))
                break


def saveActivity(env, name):
    columns = ['Close', 'Position', 'Action', 'Cash', 'Holdings', 'Money', 'Returns']
    env.data[columns].to_csv(os.path.join(RUN_DIR, name), index_label='Date')


def main():
    with open(os.path.join(RUN_DIR, 'config.json')) as f:
        config = json.load(f)
    start = time.time()
    writeStatus('running', pid=os.getpid(), started=start)

    try:
        import tradingSimulator as ts
        from tradingSimulator import TradingSimulator
        used = applyOptions(config)
        writeStatus('running', pid=os.getpid(), started=start, options=used)
        print('Options du run : ' + json.dumps(used))

        # Graine : rend le run reproductible (exploration, mélange de la mémoire de
        # rejeu, bruit d'augmentation, poids initiaux du réseau)
        if config.get('seed') is not None:
            import random
            import numpy as np
            seed = int(config['seed'])
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass
            print('Graine : {}'.format(seed))

        symbol = marketSymbol(config['stock'])
        if symbol:
            dates = [config['startingDate'], config['splitingDate'], config['endingDate']]
            validation = getattr(ts, 'validationDate', None)
            ranges = [(dates[0], dates[1]), (dates[1], dates[2])]
            if validation:
                ranges += [(dates[0], validation), (validation, dates[1])]
            prepareData(symbol, ranges)

        isTDQN = config['strategy'] == 'TDQN'
        if isTDQN:
            instrumentTDQN()
        simulator = TradingSimulator()
        extra = {}
        if config.get('transactionCosts') not in (None, ''):
            extra['transactionCosts'] = float(config['transactionCosts']) / 100
        _, trainingEnv, testingEnv = simulator.simulateNewStrategy(
            config['strategy'], config['stock'],
            startingDate=config['startingDate'],
            splitingDate=config['splitingDate'],
            endingDate=config['endingDate'],
            numberOfEpisodes=int(config['episodes']),
            # plotTraining=True : TDQN calcule alors le Sharpe à chaque épisode
            # (plus lent) et nous transmet les courbes ; les stratégies
            # classiques n'en ont pas besoin
            verbose=True, plotTraining=isTDQN, rendering=False,
            showPerformance=True, saveStrategy=False, **extra)

        saveActivity(trainingEnv, 'train.csv')
        saveActivity(testingEnv, 'test.csv')
        with open(os.path.join(RUN_DIR, 'metrics.json'), 'w') as f:
            json.dump({'train': metrics(trainingEnv.data), 'test': metrics(testingEnv.data)}, f, indent=2)
        if isTDQN:
            curves.save()
        writeStatus('done', duration=round(time.time() - start), options=used)

    except BaseException as error:
        traceback.print_exc()
        writeStatus('error', message=repr(error), duration=round(time.time() - start))
        sys.exit(1)


if __name__ == '__main__':
    main()
