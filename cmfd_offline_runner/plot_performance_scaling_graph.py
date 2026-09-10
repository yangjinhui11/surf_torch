"""3-way performance figure: Fortran 32-thread, Torch A100 eager, Torch A100
CUDA-graph (CPU series dropped).  240-step runs, repeat medians + ranges."""
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def med(path, mode):
    rep = json.load(open(path))
    out = {}
    for r in rep["runs"]:
        if r["mode"] != mode:
            continue
        out.setdefault(int(r["columns"]), []).append(float(r["integration_seconds"]))
    return {c: (statistics.median(v), min(v), max(v)) for c, v in out.items()}


eag = med("/data/yangjinhui/surf_pytorch/surf_paper/experiments/bench_final_cpu_gpu/performance_results.json", "gpu")
gph = med("/data/yangjinhui/surf_pytorch/surf_paper/experiments/bench_final_graph/performance_results.json", "gpu")
f90 = {1000: (0.55, 0.53, 0.61), 10000: (8.29, 8.27, 8.66),
       50000: (28.26, 28.14, 28.68), 97709: (53.38, 49.60, 53.57)}

cols = np.array([1000, 10000, 50000, 97709])
steps = 240


def series(dct):
    med_ = np.array([dct[int(c)][0] for c in cols])
    lo = np.array([dct[int(c)][1] for c in cols])
    hi = np.array([dct[int(c)][2] for c in cols])
    rate = cols * steps / med_
    return rate, np.vstack((rate - cols * steps / hi, cols * steps / lo - rate)), med_


f_r, f_e, f_s = series(f90)
e_r, e_e, e_s = series(eag)
g_r, g_e, g_s = series(gph)

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11})
fig, (ax_rate, ax_speed) = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
ax_rate.errorbar(cols, f_r, yerr=f_e, fmt="^-", color="#444444", lw=2, capsize=3,
                 label="Fortran reference (32 threads)")
ax_rate.errorbar(cols, e_r, yerr=e_e, fmt="s-", color="#1976a2", lw=2, capsize=3,
                 label="PyTorch A100, eager")
ax_rate.errorbar(cols, g_r, yerr=g_e, fmt="D-", color="#147a5b", lw=2, capsize=3,
                 label="PyTorch A100, CUDA graph")
ax_rate.set_xscale("log")
ax_rate.set_yscale("log")
ax_rate.set_xlabel("Number of active land columns")
ax_rate.set_ylabel("Integration throughput (column-steps s$^{-1}$)")
ax_rate.set_title(f"{steps}-step integration throughput")
ax_rate.grid(True, which="both", alpha=0.28)
ax_rate.legend(frameon=False, loc="upper left", fontsize=8)
# crossover of the CUDA-graph configuration with the Fortran reference
# (interpolate on the medians): annotate directly on the throughput panel
import numpy as _np
_c = _np.array(cols, dtype=float)
_f = _np.array([f90[int(c)][0] for c in cols])
_g = _np.array([gph[int(c)][0] for c in cols])
_x = _np.where(_g <= _f)[0]
if _x.size and _x.min() > 0:
    i0, i1 = _x.min() - 1, _x.min()
    _lg = _np.log(_g[[i0, i1]]); _lf = _np.log(_f[[i0, i1]])
    _t = (_lf[0] - _lg[0]) / (_lg[1] - _lg[0] - _lf[1] + _lf[0])
    _xc = _np.exp(_np.log(_c[i0]) + _t * (_np.log(_c[i1]) - _np.log(_c[i0])))
    ax_rate.axvline(_xc, color="0.4", linestyle=":", linewidth=1)
    ax_rate.annotate(f"crossover ~{_xc/1e3:.0f} x 10^3 columns",
                     (_xc, 4.0e4), xytext=(4, 0), textcoords="offset points",
                     fontsize=8, color="0.25", rotation=90, va="bottom")

sp_e = f_s / e_s
sp_g = f_s / g_s
ax_speed.plot(cols, sp_e, "s-", color="#1976a2", lw=2, label="eager over Fortran")
ax_speed.plot(cols, sp_g, "D-", color="#147a5b", lw=2, label="CUDA graph over Fortran")
ax_speed.axhline(1.0, color="0.35", lw=1, ls="--")
ax_speed.set_xscale("log")
ax_speed.set_xlabel("Number of active land columns")
ax_speed.set_ylabel("speed-up over the Fortran reference")
ax_speed.set_title("PyTorch configurations against the Fortran reference")
ax_speed.set_ylim(0.0, float(max(sp_e.max(), sp_g.max())) * 1.2)
ax_speed.grid(True, which="both", alpha=0.28)
ax_speed.legend(frameon=False, fontsize=8, loc="lower right")
for x, y in zip(cols, sp_g):
    ax_speed.annotate(f"{y:.2f}x", (x, y), xytext=(0, 7), textcoords="offset points",
                      ha="center", color="#147a5b", fontsize=8)

out = "/data/yangjinhui/surf_pytorch/surf_paper/manuscript/figures/surf_fortran_graph_scaling.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
