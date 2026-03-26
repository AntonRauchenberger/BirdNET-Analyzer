- Starten der python Umgebung: `source birdnet-env/bin/activate`
- Starten der Analyse: `python -m birdnet_analyzer.analyze ownTests/audios/amsel.wav`

- Erster optimierter Startvorgang:
```
source birdnet-env/bin/activate

python -c "import os; import birdnet_analyzer.config as cfg; \
cfg.BIRDNET_MODEL_PATH='checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_INT8.tflite'; \
inp='ownTests/audios/amsel.wav'; \
analyse_name=os.path.splitext(os.path.basename(inp))[0]; \
outdir=os.path.join('ownTests','results',analyse_name); \
from birdnet_analyzer.analyze.core import analyze; \
analyze(inp, \
output=outdir, \
threads=1, batch_size=1, \
min_conf=0.5, top_n=3, \
overlap=0.0, sensitivity=1.0, \
fmin=200, fmax=12000, \
merge_consecutive=3, \
rtype=['table'], \
combine_results=False, skip_existing_results=True)"
```