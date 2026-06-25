import os
import sys

_scripts_dir = os.path.dirname(__file__)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_sft

if __name__ == "__main__":
    run_sft.main()
