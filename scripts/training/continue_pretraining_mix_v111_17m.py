#!/usr/bin/env python3
"""Pilote CPT v1.1.1 prudent sur le corpus français et ivoirien audité."""

from __future__ import annotations

import continue_pretraining_mix_v10_17m as implementation


implementation.MIXTURE_ID = "ivoireslm_pretraining_mix_v1.1.1_pilot"
implementation.MODEL_ID = "microivoire_transformer_v1.4_17m_cpt_v111_pilot"
implementation.WEIGHTS = {
    "base_v09": 0.70,
    "natural_french_open": 0.20,
    "natural_french_conversation_open": 0.05,
    "natural_ivoirian_grounded_verified": 0.05,
}
implementation.MAX_STEPS = 500
implementation.LEARNING_RATE = 1e-5
implementation.MINIMUM_LEARNING_RATE = 1e-6
implementation.WARMUP_STEPS = 25
implementation.EVALUATION_INTERVAL = 125
implementation.CHECKPOINT_INTERVAL = 125
implementation.SEED = 20260903
implementation.MAXIMUM_BASE_VALIDATION_INCREASE = 0.015


if __name__ == "__main__":
    implementation.main()
