import os
import runpy
from pathlib import Path


os.environ["IVOIRESLM_CORPUS_VERSION"] = "ivoireslm_corpus_v0.4.0"
os.environ["IVOIRESLM_INCLUDE_WDI"] = "1"
os.environ["IVOIRESLM_INCLUDE_FAOSTAT"] = "1"
os.environ["IVOIRESLM_INCLUDE_OPEN_FRENCH"] = "1"
runpy.run_path(
    Path(__file__).with_name("build_corpus_v01.py"),
    run_name="__main__",
)
