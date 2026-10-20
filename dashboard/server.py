# coding=utf-8

"""
Petit site local pour lancer des runs et comparer leurs résultats.
Bibliothèque standard uniquement (Python 3.7).

Lancement, depuis la racine du dépôt, environnement tdqn activé :
    python dashboard/server.py
puis ouvrir http://localhost:8000

Chaque run est un processus séparé (run_worker.py). Ses fichiers sont dans
dashboard/runs/<id>/ : on peut donc fermer le site et le relancer sans perdre
les résultats.

Les runs passent par une file d'attente : au plus PARALLEL runs tournent en
même temps (2 par défaut, modifiable : python dashboard/server.py --parallel 3).
Les runs en attente reprennent si on relance le site.
"""

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, 'runs')
PORT = 8000
PARALLEL = 2

processes = {}   # id du run -> subprocess.Popen lancé par ce serveur
ADVANCED = ('positionMode', 'dailyShortCost', 'shortEpsilon', 'fractionalShares',
            'noiseRange', 'validationDate', 'transactionCosts')
SWEEPABLE = ADVANCED + ('episodes',)   # paramètres qu'un balayage peut faire varier
lock = threading.Lock()


def loadOptions():
    """Stratégies, actifs et dates par défaut, lus dans tradingSimulator.py."""
    sys.path.insert(0, ROOT)
    import tradingSimulator as ts
    import tradingEnv
    import dataAugmentation
    stocks = []
    for name in ('cryptos', 'indices', 'companies', 'fictives'):
        stocks += list(getattr(ts, name, {}).keys())
    # Options des étapes 3 à 6 : valeurs par défaut lues dans le code, None si
    # l'option n'existe pas (le formulaire la masque alors)
    epsilon = getattr(tradingEnv, 'shortEpsilon', None)
    noise = getattr(dataAugmentation, 'noiseRange', None)
    advanced = {
        'positionModes': list(getattr(tradingEnv, 'positionModes', ())),
        'positionMode': getattr(tradingEnv, 'positionMode', None),
        'dailyShortCost': getattr(tradingEnv, 'dailyShortCost', None),
        'shortEpsilon': None if epsilon is None else str(epsilon),
        'fractionalShares': getattr(tradingEnv, 'fractionalShares', None),
        'noiseRange': None if noise is None else ', '.join(str(x) for x in noise),
        'validationDate': getattr(ts, 'validationDate', None),
        'transactionCosts': round(100 * ts.transactionCosts, 4),
    }
    return dict(advanced, **{
        'strategies': list(ts.strategiesAI.keys()) + list(ts.strategies.keys()),
        'stocks': stocks,
        'startingDate': ts.startingDate,
        'splitingDate': ts.splitingDate,
        'endingDate': ts.endingDate,
        'episodes': ts.numberOfEpisodes,
    })


def readJSON(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def runStatus(runId):
    status = readJSON(os.path.join(RUNS, runId, 'status.json'), {'state': 'pending'})
    process = processes.get(runId)
    # Processus mort sans avoir écrit son état final (plantage, arrêt manuel)
    if status['state'] in ('pending', 'running') and process is not None and process.poll() is not None:
        status = {'state': 'error', 'message': 'processus terminé (code {})'.format(process.returncode)}
    # Run laissé "running" par une session précédente du serveur
    if status['state'] == 'running' and process is None:
        status = {'state': 'unknown', 'message': 'serveur redémarré pendant le run'}
    if status['state'] == 'running':
        progress = readProgress(runId)
        if progress and status.get('started'):
            # temps écoulé réel (tqdm ne se met à jour qu'à la fin d'un épisode)
            # et temps restant extrapolé à partir des épisodes déjà faits
            elapsed = time.time() - status['started']
            progress['elapsed'] = round(elapsed)
            done, total = progress['done'], progress['total']
            progress['remaining'] = round(elapsed / done * (total - done)) if done else None
        status['progress'] = progress
    return status


def readProgress(runId):
    """
    Dernière ligne de tqdm dans log.txt, par exemple
    ' 33%|███▎      | 1/3 [00:23<00:47, 23.50s/it]' : épisodes faits et total.
    """
    try:
        with open(os.path.join(RUNS, runId, 'log.txt'), 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            tail = f.read().decode('utf-8', errors='replace')
    except OSError:
        return None
    matches = re.findall(r'(\d+)/(\d+) \[([\d:]+)<([\d:?]+)', tail)
    if not matches:
        return None
    done, total = matches[-1][:2]
    return {'done': int(done), 'total': int(total)}


def listRuns():
    if not os.path.isdir(RUNS):
        return []
    runs = []
    for runId in sorted(os.listdir(RUNS), reverse=True):
        folder = os.path.join(RUNS, runId)
        config = readJSON(os.path.join(folder, 'config.json'))
        if config is None:
            continue
        runs.append({'id': runId, 'config': config, 'status': runStatus(runId),
                     'metrics': readJSON(os.path.join(folder, 'metrics.json'))})
    return runs


def readActivity(path):
    """CSV d'activité -> colonnes en listes (pour les graphiques)."""
    if not os.path.exists(path):
        return None
    series = {'Date': [], 'Close': [], 'Money': [], 'Action': [], 'Position': []}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            series['Date'].append(row['Date'][:10])
            for key in ('Close', 'Money', 'Action', 'Position'):
                series[key].append(float(row[key]))
    return series


def writeJSON(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def createRun(config):
    """Crée le dossier du run et le met dans la file d'attente."""
    label = re.sub(r'[^A-Za-z0-9_.-]+', '-', config.get('label') or config['strategy'])[:60]
    runId = time.strftime('%Y%m%d-%H%M%S') + '_' + label
    folder = os.path.join(RUNS, runId)
    suffix = 1
    while os.path.exists(folder):   # deux runs créés dans la même seconde
        suffix += 1
        folder = os.path.join(RUNS, '{}-{}'.format(runId, suffix))
    os.makedirs(folder)
    writeJSON(os.path.join(folder, 'config.json'), config)
    writeJSON(os.path.join(folder, 'status.json'), {'state': 'queued', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                    'queuedAt': time.time()})
    return os.path.basename(folder)


def scheduler():
    """Lance les runs en attente, du plus ancien au plus récent, sans dépasser PARALLEL."""
    while True:
        with lock:
            running = sum(1 for p in processes.values() if p.poll() is None)
            if running < PARALLEL and os.path.isdir(RUNS):
                queued = []
                for runId in os.listdir(RUNS):
                    status = readJSON(os.path.join(RUNS, runId, 'status.json'), {})
                    if status.get('state') == 'queued':
                        queued.append((status.get('queuedAt', 0), runId))
                for _, runId in sorted(queued)[:PARALLEL - running]:
                    startRun(runId)
        time.sleep(1)


def startRun(runId):
    folder = os.path.join(RUNS, runId)
    writeJSON(os.path.join(folder, 'status.json'), {'state': 'pending', 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    log = open(os.path.join(folder, 'log.txt'), 'w')
    # PyTorch prend par défaut tous les cœurs : avec plusieurs runs en parallèle,
    # ils se gênent. On partage les cœurs entre les runs.
    threads = str(max(1, (os.cpu_count() or 1) // PARALLEL))
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    processes[runId] = subprocess.Popen(
        [sys.executable, '-u', os.path.join(HERE, 'run_worker.py'), folder],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
        # Windows : Ctrl+C dans le terminal du site ne doit pas interrompre les runs
        creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))


def parseValue(key, text):
    """Valeur d'un balayage (texte du formulaire) -> valeur de config."""
    text = text.strip()
    if key == 'fractionalShares':
        return text.lower() in ('1', 'true', 'oui', 'yes', 'on')
    return text


def buildConfigs(body):
    """
    Config de base + répétitions (graines) + balayage d'un paramètre.
    Renvoie la liste des configs à créer, toutes avec le même groupe.
    """
    base = {key: body.get(key) or OPTIONS.get(key) for key in
            ('strategy', 'stock', 'startingDate', 'splitingDate', 'endingDate', 'episodes')}
    base['label'] = body.get('label') or ''
    # Options avancées : seulement celles que le code du dépôt connaît
    for key in ADVANCED:
        if OPTIONS.get(key) is None:
            continue
        value = body.get(key)
        if key == 'fractionalShares':
            value = bool(value)
        base[key] = OPTIONS[key] if value in (None, '') else value

    # Balayage : valeurs séparées par « ; » (la virgule sert déjà dans le bruit)
    sweepKey = body.get('sweepKey') or ''
    values = [None]
    if sweepKey in SWEEPABLE and (OPTIONS.get(sweepKey) is not None or sweepKey == 'episodes'):
        values = [v for v in (body.get('sweepValues') or '').split(';') if v.strip()] or [None]
    else:
        sweepKey = ''

    # Répétitions : graines seed, seed+1, ... ; sans graine ni répétition, pas de graine
    repeats = max(1, min(50, int(body.get('repeats') or 1)))
    seed = str(body.get('seed') or '').strip()
    seeds = [None] if repeats == 1 and not seed else [int(seed or 1) + k for k in range(repeats)]

    group = time.strftime('%Y%m%d-%H%M%S') if len(values) * len(seeds) > 1 else None
    name = base['label'] or base['strategy'].replace(' ', '')
    configs = []
    for value in values:
        for s in seeds:
            config = dict(base)
            parts = [name]
            if value is not None:
                config[sweepKey] = parseValue(sweepKey, value)
                config['variant'] = '{}={}'.format(sweepKey, value.strip())
                parts.append(re.sub(r'\s+', '', value))
            if s is not None:
                config['seed'] = s
                parts.append('s{}'.format(s))
            if group:
                config['group'] = group
                config['groupLabel'] = name
            config['label'] = '_'.join(parts)
            configs.append(config)
    return configs


class Handler(BaseHTTPRequestHandler):

    def send(self, body, contentType='application/json', code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode('utf-8') if contentType == 'application/json' else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', contentType + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def runFolder(self, runId):
        folder = os.path.join(RUNS, os.path.basename(runId))
        return folder if os.path.isdir(folder) else None

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            with open(os.path.join(HERE, 'index.html'), 'rb') as f:
                return self.send(f.read(), 'text/html')
        if path == '/api/options':
            return self.send(OPTIONS)
        if path == '/api/runs':
            return self.send(listRuns())
        match = re.fullmatch(r'/api/runs/([^/]+)(/log)?', path)
        if match:
            folder = self.runFolder(match.group(1))
            if folder is None:
                return self.send({'error': 'run inconnu'}, code=404)
            if match.group(2):
                try:
                    with open(os.path.join(folder, 'log.txt'), encoding='utf-8', errors='replace') as f:
                        text = f.read()
                except OSError:
                    text = ''
                # tqdm réécrit sa ligne avec \r : on ne garde que le dernier état
                text = '\n'.join(line.split('\r')[-1] for line in text.split('\n'))
                return self.send(text[-20000:], 'text/plain')
            runId = os.path.basename(folder)
            return self.send({
                'id': runId,
                'config': readJSON(os.path.join(folder, 'config.json')),
                'status': runStatus(runId),
                'metrics': readJSON(os.path.join(folder, 'metrics.json')),
                'train': readActivity(os.path.join(folder, 'train.csv')),
                'test': readActivity(os.path.join(folder, 'test.csv')),
                'curves': readJSON(os.path.join(folder, 'curves.json')),
            })
        self.send({'error': 'introuvable'}, code=404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        if self.path == '/api/runs':
            try:
                configs = buildConfigs(body)
            except ValueError as error:
                return self.send({'error': str(error)}, code=400)
            with lock:
                ids = [createRun(config) for config in configs]
            return self.send({'ids': ids})
        match = re.fullmatch(r'/api/runs/([^/]+)/stop', self.path)
        if match:
            runId = os.path.basename(match.group(1))
            with lock:
                status = readJSON(os.path.join(RUNS, runId, 'status.json'), {})
                if status.get('state') == 'queued':
                    writeJSON(os.path.join(RUNS, runId, 'status.json'), {'state': 'stopped', 'message': 'retiré de la file'})
                    return self.send({'stopped': True})
                if runId in processes:
                    processes[runId].kill()
                    processes[runId].wait()
                    writeJSON(os.path.join(RUNS, runId, 'status.json'), {'state': 'stopped', 'message': 'arrêté depuis le site'})
                    return self.send({'stopped': True})
        self.send({'error': 'introuvable'}, code=404)

    def log_message(self, *args):
        pass   # pas de ligne dans la console à chaque requête


if __name__ == '__main__':
    os.makedirs(RUNS, exist_ok=True)
    print('Chargement du code du dépôt...')
    OPTIONS = loadOptions()
    if '--parallel' in sys.argv:
        PARALLEL = max(1, int(sys.argv[sys.argv.index('--parallel') + 1]))
    threading.Thread(target=scheduler, daemon=True).start()
    print('Site disponible sur http://localhost:{} ({} runs en parallèle au plus)'.format(PORT, PARALLEL))
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
