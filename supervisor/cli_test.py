from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from supervisor.utils.config import (
    DRFConfig, Context, SupervisorConfig, SystemConfig
)
from supervisor.adapter.drf_io import DigitalRFLoader
from supervisor.adapter.drf_features import extract_features, stack_for_model, FEATURE_NAMES
from supervisor.core.supervisor import Supervisor

# --- Build one config object for the entire system ---
sys_cfg = SystemConfig(
    drf=DRFConfig(
        root=Path(r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol"),
        channel=None,
        ns_read=262_144,
        win=4096,
        hop=2048,
        max_windows=None
    ),
    ctx=Context(
        sample_rate=2_500_000.0,
        epoch_unix=0.0
    ),
    sup=SupervisorConfig(
        warn=0.35,
        fail=0.65,
        warmup_windows=200,
        w_cos=0.6,
        w_z=0.4,
        agg="max",
        center_freq_hz=1_024_000_000.4842604
    )
)

# --- Adapter + Supervisor ---
ldr = DigitalRFLoader(sys_cfg.drf, sys_cfg.ctx)
sup = Supervisor.from_system_config(sys_cfg, feature_names=FEATURE_NAMES)

print("META:", ldr.meta)

for x, ks, ke in ldr.stream_windows(sys_cfg.drf.win, sys_cfg.drf.hop, normalize_power=True):
    row = sup.process_window(
        x_win=x,
        k_start=ks,
        k_end=ke,
        extract_features=extract_features,
        stack_for_model=stack_for_model,
        k_to_unix=ldr.k_to_unix,
        sample_rate_hz=sys_cfg.ctx.sample_rate,
    )
    print(json.dumps(row))
