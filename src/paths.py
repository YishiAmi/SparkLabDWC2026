"""Single place for all paths. Everything is relative to the project folder, so the code runs anywhere.
Set DWC_RAW to the folder that contains the organisers' 'TestCases' folder (only needed for steps 0 and grid checks)."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"; MODELS=ROOT/"models"; OUT=ROOT/"outputs"; GRID=DATA/"grid"
RAW=Path(os.environ.get("DWC_RAW",ROOT/"raw"))
for d in (DATA,MODELS,OUT): d.mkdir(exist_ok=True)
