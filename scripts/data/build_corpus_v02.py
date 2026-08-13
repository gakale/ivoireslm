import os
import runpy
from pathlib import Path


os.environ["IVOIRESLM_CORPUS_VERSION"] = "ivoireslm_corpus_v0.2.0"
os.environ["IVOIRESLM_INCLUDE_WDI"] = "1"
runpy.run_path(
    Path(__file__).with_name("build_corpus_v01.py"),
    run_name="__main__",
)
