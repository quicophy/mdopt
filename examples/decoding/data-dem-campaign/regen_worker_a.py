"""Regeneration worker A: the d=9 near-threshold cells (the slow block).

16000 shots per cell to match the topped-up final state; the per-cell RNG
stream is seeded exactly as the original campaign, so the records are
bit-identical to the purged data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nishimori

nishimori.RESULTS.mkdir(exist_ok=True)
for p in (0.095, 0.100, 0.104, 0.108, 0.112, 0.116):
    nishimori.run(9, p, shots=16000)
print("WORKER A DONE", flush=True)
