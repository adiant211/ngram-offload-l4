"""Full-size end-of-step sparse update, CPU only: 976,560 rows x 512, 816k touched."""
import os, sys, time, types
sys.path.insert(0, sys.argv[1])
import torch
import pinned_ngram as P
rows_total, dim, touched_rows = 976_560, 512, 816_000
cfg = types.SimpleNamespace(beta2=0.99, eps=1e-8, lr=0.02, dim=dim)
torch.manual_seed(0)
m = types.SimpleNamespace(cfg=cfg, _pin_detail={}, _ram_profile=dict(update=0.0), _pin_slots=[])
m.table = torch.randn(rows_total, dim).mul_(0.02).to(torch.bfloat16)
m.row_v = torch.rand(rows_total).mul_(1e-4)
m._host_grad = torch.zeros(rows_total, dim)
m._host_touched = torch.zeros(rows_total, dtype=torch.bool)
pick = torch.randperm(rows_total)[:touched_rows]
fresh = torch.randn(touched_rows, dim).mul_(1e-3)
cores = os.cpu_count()
print("cores", cores, flush=True)
for mode, threads, block in [("rows", 4, 4096), ("rows", cores, 4096), ("blocks", 4, 4096),
                             ("blocks", cores, 4096), ("blocks", cores, 1024), ("blocks", cores, 16384)]:
    m._sparse_update, m._update_threads, m._update_block = mode, threads, block
    torch.set_num_threads(4)
    times = []
    for _ in range(3):
        m._host_grad[pick] = fresh
        m._host_touched[pick] = True
        t = time.perf_counter()
        n = P._staged_step(m)
        times.append(time.perf_counter()-t)
        assert n == touched_rows and not m._host_touched.any() and not m._host_grad[:4096].any()
    print("UPDATE", mode, "threads", threads, "block", block, "seconds", [round(x, 3) for x in times], flush=True)
