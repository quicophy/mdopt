"""Regeneration worker B: everything except d=9 near-threshold.

d=5 and d=7 at 16000 shots, d=11 at 4000, and the three wing rates at 4000
for all distances -- the same shot counts the campaign ended with.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nishimori

nishimori.RESULTS.mkdir(exist_ok=True)
for distance in (5, 7):
    for p in (0.095, 0.100, 0.104, 0.108, 0.112, 0.116):
        nishimori.run(distance, p, shots=16000)
for p in (0.095, 0.100, 0.104, 0.108, 0.112, 0.116):
    nishimori.run(11, p, shots=4000)
for p in (0.070, 0.085, 0.130):
    for distance in (5, 7, 9, 11):
        nishimori.run(distance, p, shots=4000)
print("WORKER B DONE", flush=True)
