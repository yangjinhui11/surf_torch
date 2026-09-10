#!/usr/bin/env python3
"""Synthetic-twin calibration of one SURF vegetation parameter."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from surf_pytorch import IdealizedDiurnalConfig, LandSurface, SurfConfig, build_idealized_diurnal


OUTDIR = Path(os.environ.get("SURF_TRAINING_OUTPUT", "."))
DTYPE = torch.float64
N_STEPS = int(os.environ.get("SURF_TRAINING_STEPS", "24"))
EPOCHS = int(os.environ.get("SURF_TRAINING_EPOCHS", "20"))
TRUTH_LAI = float(os.environ.get("SURF_TRUTH_LAI", "3.5"))
INITIAL_LAI = float(os.environ.get("SURF_INITIAL_LAI", "7.0"))
LEARNING_RATE = float(os.environ.get("SURF_TRAINING_LR", "0.15"))
VEG_CLASS = int(os.environ.get("SURF_TRAINING_VEG_CLASS", "3"))
DEVICE = os.environ.get("SURF_DEVICE", "cpu")


def rollout(model, forcing, static, initial_state):
    state = initial_state
    evaporation = torch.zeros((), dtype=DTYPE, device=state.skin_t.device)
    for item in forcing:
        state, diag = model(state, item, static)
        evaporation = evaporation + diag.e_flux.mean() * item.dt.mean()
    fraction = model._tile_frac(static, model.config.n_tile, state=state)
    terminal_skin = (state.skin_t * fraction).sum(dim=1).mean()
    return terminal_skin, evaporation


def build_case():
    config = IdealizedDiurnalConfig(
        n_grid=4,
        n_steps=N_STEPS,
        dt_seconds=3600.0,
        lat_deg=45.0,
        lon_deg=0.0,
        day_of_year=172,
        tair_mean=290.0,
        tair_amp=8.0,
        sw_peak=800.0,
        soil_t_init=285.0,
        soil_w_init=0.30,
        skin_t_init=288.0,
        dtype=DTYPE,
        device=DEVICE,
    )
    return build_idealized_diurnal(config)


def make_model() -> LandSurface:
    return LandSurface(SurfConfig(
        dtype=DTYPE,
        differentiable=True,
        validate=False,
        learnable_veg=("rvlai",),
        device=DEVICE,
    ))


def main() -> None:
    forcing, static, initial_state = build_case()

    truth_model = make_model()
    with torch.no_grad():
        truth_model.veg_table.rvlai[VEG_CLASS] = TRUTH_LAI
    with torch.no_grad():
        target_skin, target_evap = rollout(
            truth_model, forcing, static, initial_state,
        )

    model = make_model()
    with torch.no_grad():
        model.veg_table.rvlai[VEG_CLASS] = INITIAL_LAI
    parameter = model.veg_table.rvlai
    optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)

    rows = []
    for epoch in range(EPOCHS + 1):
        skin, evap = rollout(model, forcing, static, initial_state)
        # The scales keep the temperature and water losses numerically
        # comparable while preserving their physical interpretation.
        skin_error = skin - target_skin.detach()
        evap_error = (evap - target_evap.detach()) / 0.1
        loss = skin_error.square() + evap_error.square()
        gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
        rows.append({
            "epoch": epoch,
            "lai": float(parameter[VEG_CLASS].detach()),
            "loss": float(loss.detach()),
            "skin_error_K": float(skin_error.detach()),
            "evap_error_scaled": float(evap_error.detach()),
            "gradient_class": float(gradient[VEG_CLASS].detach()),
        })
        if epoch == EPOCHS:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(min=0.1)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "synthetic_training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_steps": N_STEPS,
        "epochs": EPOCHS,
        "vegetation_class": VEG_CLASS,
        "truth_lai": TRUTH_LAI,
        "initial_lai": INITIAL_LAI,
        "final_lai": rows[-1]["lai"],
        "final_loss": rows[-1]["loss"],
        "target_skin_K": float(target_skin),
        "target_evaporation_kg_m2": float(target_evap),
        "finite": all(torch.isfinite(value).item() for value in (target_skin, target_evap)),
    }
    (OUTDIR / "synthetic_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["lai"] for row in rows], "o-", color="#1976a2")
    axes[0].axhline(TRUTH_LAI, color="#c43c39", ls="--", label="truth")
    axes[0].set(xlabel="Optimization epoch", ylabel="LAI", title="Parameter recovery")
    axes[0].legend(frameon=False)
    axes[1].semilogy(epochs, [max(row["loss"], 1.0e-16) for row in rows], "o-", color="#54a24b")
    axes[1].set(xlabel="Optimization epoch", ylabel="Weighted loss", title="Synthetic-twin objective")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.savefig(OUTDIR / "synthetic_training_curve.png", dpi=300, bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
