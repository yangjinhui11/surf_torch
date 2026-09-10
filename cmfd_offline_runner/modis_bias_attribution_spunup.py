#!/usr/bin/env python3
"""Attribution diagnostics for the daytime MODIS LST cold bias (2018).

Reads the matched observation/model/view-time triple produced by the MODIS
validation experiment plus the CMFD forcing, and decomposes the strict-mask
daytime cold bias into diurnal-amplitude, surface-air-contrast, regional,
forcing-transmissivity, and model-state components.
"""
import json

import numpy as np
from netCDF4 import Dataset

OBS = "/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg/modis_cmg_2018_cmfd_columns.nc"
MOD = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
       "modis_validation_1y_2018_spunup_20260822/surf_modis_observables.nc")
STATE = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
         "modis_validation_1y_2018_spunup_20260822/torch_offline_daily.npz")
DATA = "/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset"
OUT = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
       "modis_validation_1y_2018_spunup_20260822/comparison/modis_bias_attribution.json")

obs = Dataset(OBS)
mod = Dataset(MOD)
lat = np.asarray(obs["lat"][:], np.float64)
lon = np.asarray(obs["lon"][:], np.float64)
N = lat.size
od = np.asarray(obs["lst_day"][:], np.float64)
on = np.asarray(obs["lst_night"][:], np.float64)
vd = np.asarray(obs["lst_day_view_time_utc"][:], np.float64)
sd = np.asarray(obs["lst_day_strict"][:], bool)
sn = np.asarray(obs["lst_night_strict"][:], bool)
osf = np.asarray(obs["snow_fraction"][:], np.float64)
md = np.asarray(mod["lst_day"][:], np.float64)
mn = np.asarray(mod["lst_night"][:], np.float64)
msf = np.asarray(mod["snow_fraction"][:], np.float64)

tair = np.load(DATA + "/tair_K.npy", mmap_mode="r")
swdn = np.load(DATA + "/swdown_W_m2.npy", mmap_mode="r")
cols = np.arange(N)


def sample_day(field, day):
    h = day * 24.0 + vd[day]
    j = np.clip((h / 3.0).astype(np.int64), 0, 2918)
    frac = np.clip((h - 3.0 * j) / 3.0, 0.0, 1.0)
    a = np.asarray(field[j, cols], np.float64)
    b = np.asarray(field[j + 1, cols], np.float64)
    return a * (1.0 - frac) + b * frac


doy = np.arange(365)
decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284.0 + doy + 1) / 365.0)
month_bounds = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
month_of = np.clip(np.searchsorted(month_bounds, np.arange(365), side="right") - 1, 0, 11)

tp = (lat >= 27) & (lat <= 39) & (lon >= 73) & (lon <= 104)
arid = (lat >= 36) & (lat <= 50) & (lon >= 74) & (lon <= 120) & ~tp
humid = (lat >= 20) & (lat <= 35) & (lon >= 105) & (lon <= 135) & ~tp
ne = (lat >= 40) & (lat <= 54) & (lon >= 119) & (lon <= 135)
regions = {"tibetan_plateau": tp, "arid_northwest": arid,
           "humid_southeast": humid, "northeast": ne}

R = {"n_columns": int(N)}
w_cos = np.cos(np.deg2rad(lat))

tv_all = [sample_day(tair, day) for day in range(365)]
sv_all = [sample_day(swdn, day) for day in range(365)]

monthly = []
for m in range(12):
    days = doy[month_of == m]
    row = {"month": m + 1}
    for tag, o, mo_, smask in (("day", od, md, sd), ("night", on, mn, sn)):
        valid = smask[days] & np.isfinite(o[days]) & np.isfinite(mo_[days])
        diff = np.where(valid, mo_[days] - o[days], 0.0)
        wv = w_cos[None, :] * valid
        bias = float(np.sum(wv * diff) / np.sum(wv))
        rm = float(np.sqrt(np.sum(wv * diff ** 2) / np.sum(wv)))
        row[tag + "_bias"], row[tag + "_rmse"], row[tag + "_n"] = bias, rm, int(valid.sum())
    monthly.append(row)
R["monthly"] = monthly

snowfree = (msf < 1e-3) & (osf < 0.5)
res = {k: [] for k in ("c_obs", "c_mod", "bias", "tau", "amp_obs", "amp_mod")}
reg_acc = {k: {"bias": [], "c_obs": [], "c_mod": [], "w": []} for k in regions}
for day in range(365):
    m = sd[day] & snowfree[day] & np.isfinite(od[day]) & np.isfinite(md[day])
    if not m.any():
        continue
    tv = tv_all[day]
    h = vd[day]
    H = np.deg2rad(15.0 * (h + lon / 15.0 - 12.0))
    cz = (np.sin(np.deg2rad(lat)) * np.sin(decl[day])
          + np.cos(np.deg2rad(lat)) * np.cos(decl[day]) * np.cos(H))
    tau = np.where(cz > 0.15, sv_all[day] / np.maximum(1361.0 * cz, 1e-6), np.nan)
    bias = md[day] - od[day]
    for k in regions:
        mk = m & regions[k]
        if mk.any():
            reg_acc[k]["bias"].append(bias[mk])
            reg_acc[k]["c_obs"].append((od[day] - tv)[mk])
            reg_acc[k]["c_mod"].append((md[day] - tv)[mk])
            reg_acc[k]["w"].append(w_cos[mk])
    res["c_obs"].append((od[day] - tv)[m])
    res["c_mod"].append((md[day] - tv)[m])
    res["bias"].append(bias[m])
    res["tau"].append(tau[m])
    mnk = sn[day] & snowfree[day] & np.isfinite(on[day]) & np.isfinite(mn[day]) & m
    if mnk.any():
        res["amp_obs"].append((od[day] - on[day])[mnk])
        res["amp_mod"].append((md[day] - mn[day])[mnk])

cat = {k: np.concatenate(v) for k, v in res.items() if v}
amp_obs = np.concatenate(res["amp_obs"])
amp_mod = np.concatenate(res["amp_mod"])
R["day_strict_snowfree"] = {
    "n": int(cat["bias"].size),
    "bias_mean": float(cat["bias"].mean()),
    "c_obs_mean": float(cat["c_obs"].mean()),
    "c_mod_mean": float(cat["c_mod"].mean()),
    "amp_obs_mean": float(amp_obs.mean()),
    "amp_mod_mean": float(amp_mod.mean()),
    "amp_obs_median": float(np.median(amp_obs)),
    "amp_mod_median": float(np.median(amp_mod)),
    "tau_mean": float(np.nanmean(cat["tau"])),
    "tau_p10": float(np.nanpercentile(cat["tau"], 10)),
    "tau_p50": float(np.nanpercentile(cat["tau"], 50)),
    "tau_p90": float(np.nanpercentile(cat["tau"], 90)),
    "frac_tau_below_0.4": float(np.nanmean(cat["tau"] < 0.4)),
}

co = cat["c_obs"]
b = cat["bias"]
ta = np.nan_to_num(cat["tau"], nan=-1.0)


def decile_means(x, y):
    qs = np.quantile(x, np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        sel = (x >= lo) & (x < hi) if i < 9 else (x >= lo) & (x <= hi)
        out.append(float(y[sel].mean()))
    return out, [float(q) for q in qs]


bm, qs = decile_means(co, b)
R["bias_by_contrast_decile"] = bm
R["contrast_decile_edges"] = qs
bt, qt = decile_means(ta, b)
R["bias_by_tau_decile"] = bt
R["tau_decile_edges"] = qt

R["regional"] = {}
for k in regions:
    if reg_acc[k]["bias"]:
        bb = np.concatenate(reg_acc[k]["bias"])
        co_ = np.concatenate(reg_acc[k]["c_obs"])
        cm_ = np.concatenate(reg_acc[k]["c_mod"])
        ww = np.concatenate(reg_acc[k]["w"])
        R["regional"][k] = {"n": int(bb.size),
                            "bias_w": float(np.sum(ww * bb) / np.sum(ww)),
                            "c_obs": float(co_.mean()),
                            "c_mod": float(cm_.mean())}

z = np.load(STATE)
sw1 = z["soil_w"][:, :, 0].mean(axis=0)
sw4 = z["soil_w"][:, :, 3].mean(axis=0)
pcol_bias = np.zeros(N)
pcol_n = np.zeros(N)
for day in range(365):
    m = sd[day] & snowfree[day] & np.isfinite(od[day]) & np.isfinite(md[day])
    pcol_bias += np.where(m, md[day] - od[day], 0.0)
    pcol_n += m
ok = pcol_n > 30
pb = pcol_bias[ok] / pcol_n[ok]


def corr(x):
    return float(np.corrcoef(x[ok], pb)[0, 1])


R["state_link"] = {"n_columns": int(ok.sum()),
                   "corr_bias_soilw_l1": corr(sw1),
                   "corr_bias_soilw_l4": corr(sw4)}

with open(OUT, "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps({k: R[k] for k in ("day_strict_snowfree", "regional", "state_link")},
                 indent=1))
print("monthly day bias:", [round(r["day_bias"], 2) for r in R["monthly"]])
print("monthly day rmse:", [round(r["day_rmse"], 2) for r in R["monthly"]])
print("bias_by_contrast_decile:", [round(x, 2) for x in R["bias_by_contrast_decile"]])
print("contrast_decile_edges:", [round(x, 1) for x in R["contrast_decile_edges"]])
print("bias_by_tau_decile:", [round(x, 2) for x in R["bias_by_tau_decile"]])
print("tau_decile_edges:", [round(x, 2) for x in R["tau_decile_edges"]])
