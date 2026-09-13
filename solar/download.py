"""Download original data; never silently substitute synthetic measurements."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://www.kaggle.com/api/v1/datasets/download/anikannal/solar-power-generation-data'
FILES = ['Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv']

def extract(payload, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in FILES:
            (destination / name).write_bytes(archive.read(name))
    manifest = {'source': URL, 'archive_sha256': hashlib.sha256(payload).hexdigest(),
                'files': {n: hashlib.sha256((destination / n).read_bytes()).hexdigest() for n in FILES}}
    (destination / 'provenance.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    try:
        with urllib.request.urlopen(URL, timeout=90) as response:
            extract(response.read(), 'data/raw')
        print('Downloaded Plant 1 data to data/raw')
    except Exception as exc:
        raise SystemExit('Download failed. Download the two Plant 1 CSVs from Kaggle into data/raw. ' + str(exc))
