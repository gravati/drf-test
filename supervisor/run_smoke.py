from pathlib import Path 
from supervisor.utils.config import DRFConfig, Context
from supervisor.adapter.drf_io import DigitalRFLoader
from supervisor.utils.windows import window_iter, unit_power

ROOT = r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol"

cfg = DRFConfig(root=ROOT, win=4096, hop=2048, ns_read=262_144, max_windows=10)
ctx = Context(sample_rate=2_500_000.0, epoch_unix=0.0)

ldr = DigitalRFLoader(cfg, ctx)
print("META:", ldr.meta)

count = 0
for w, ks, ke in ldr.stream_windows(cfg.win, cfg.hop, normalize_power=True):
    t0 = float(ldr.k_to_unix(ks))
    t1 = float(ldr.k_to_unix(ke))
    print(f"window {count:04d}: k=[{ks},{ke})  t=[{t0:.6f},{t1:.6f}] s  len={len(w)}")

    count += 1

print("done.")