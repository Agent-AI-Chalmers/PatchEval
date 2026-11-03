"""Write both streams before signalling readiness for the runner timeout test."""

import sys
import time
from pathlib import Path

output, marker, delay = sys.argv[1:]
sys.stdout.write(output)
sys.stdout.flush()
sys.stderr.write("diagnostic\n")
sys.stderr.flush()
Path(marker).touch()
time.sleep(float(delay))
