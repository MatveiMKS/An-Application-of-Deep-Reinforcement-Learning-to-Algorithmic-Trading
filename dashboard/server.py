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
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, 'runs')
PORT = 8000

processes = {}   # id du run -> subprocess.Popen lancé par ce serveur


def loadOptions():
    """Stratégies, actifs et dates par défaut, lus dans tradingSimulator.py."""
    sys.path.insert(0, ROOT)
    import tradingSimulator as ts
    stocks = []
    for name in ('cryptos', 'indices', 'companies', 'fictives'):
        stocks += list(getattr(ts, name, {}).keys())
    return {
        'strategies': list(ts.strategiesAI.keys()) + list(ts.strategies.keys()),
        'stocks': stocks,
        'startingDate': ts.startingDate,
        'splitingDate': ts.splitingDate,
        'endingDate': ts.endingDate,
        'episodes': ts.numberOfEpisodes,
    }


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
    return status


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


def startRun(config):
    label = re.sub(r'[^A-Za-z0-9_-]+', '-', config.get('label') or config['strategy'])[:40]
    runId = time.strftime('%Y%m%d-%H%M%S') + '_' + label
    folder = os.path.join(RUNS, runId)
    suffix = 1
    while os.path.exists(folder):   # deux runs lancés dans la même seconde
        suffix += 1
        folder = os.path.join(RUNS, '{}-{}'.format(runId, suffix))
    runId = os.path.basename(folder)
    os.makedirs(folder)
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    log = open(os.path.join(folder, 'log.txt'), 'w')
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    processes[runId] = subprocess.Popen(
        [sys.executable, '-u', os.path.join(HERE, 'run_worker.py'), folder],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return runId


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
            config = {key: body.get(key) or OPTIONS.get(key) for key in
                      ('strategy', 'stock', 'startingDate', 'splitingDate', 'endingDate', 'episodes')}
            config['label'] = body.get('label') or ''
            return self.send({'id': startRun(config)})
        match = re.fullmatch(r'/api/runs/([^/]+)/stop', self.path)
        if match and match.group(1) in processes:
            processes[match.group(1)].kill()
            return self.send({'stopped': True})
        self.send({'error': 'introuvable'}, code=404)

    def log_message(self, *args):
        pass   # pas de ligne dans la console à chaque requête


if __name__ == '__main__':
    os.makedirs(RUNS, exist_ok=True)
    print('Chargement du code du dépôt...')
    OPTIONS = loadOptions()
    print('Site disponible sur http://localhost:{}'.format(PORT))
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
