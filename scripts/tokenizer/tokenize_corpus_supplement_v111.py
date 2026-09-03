#!/usr/bin/env python3
"""Encode le supplément v1.1.1 avec le BPE historique v0.4."""

from __future__ import annotations

import tokenize_corpus_supplement_v10 as implementation


implementation.DATASET_ID = "ivoireslm_corpus_supplement_v1.1.1"
implementation.TOKENIZED_ID = "ivoireslm_corpus_supplement_v1.1.1_bpe_v0.4"
implementation.BUCKETS = {
    "wikipedia_fr_natural_v0.3": "natural_french_open",
    "openassistant_fr_v0.1": "natural_french_conversation_open",
    "data_gouv_ci_open_v0.1.1": "natural_ivoirian_grounded_verified",
}


if __name__ == "__main__":
    implementation.main()
