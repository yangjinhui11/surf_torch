# Determinism / provenance probes (2026-08-22)

Question: why does the decade chain rerun (launched 19:24 UTC, with the
SURFTSTP permanent-snow reset) differ from the archived `_prefix` run
(launched 06:26 UTC) at round-off level on day 10, amplifying to ~1e-3 rms
SWE by year end?

## Probes (30-day chained 2009 replays from the shared restart, exact
production loop; `determinism_probe.py`)

| file                     | code            | interpreter | GPU    | result |
|--------------------------|-----------------|-------------|--------|--------|
| ab_a6000_active.npz      | canonical+psn   | yjh torch 2.11.0+cu130 | A6000 | bit-identical to ab_a100_active |
| ab_a6000_neutral.npz     | zsn_pert=1e30   | yjh torch 2.11.0+cu130 | A6000 | bit-identical to active (fix is a no-op below 9000 kg m-2) |
| ab_a100_active.npz       | canonical+psn   | yjh torch 2.11.0+cu130 | A100  | bit-identical to the new chain's day-30 record |
| ab_a100_neutral.npz      | zsn_pert=1e30   | yjh torch 2.11.0+cu130 | A100  | bit-identical |
| ab_a100_noguard.npz      | minus ZCVW guard| yjh torch 2.11.0+cu130 | A100  | bit-identical (guard never fires in 30 d) |
| ab_a100_snapshot.npz     | Aug-9 snapshot  | yjh torch 2.11.0+cu130 | A100  | bit-identical |
| ab_a100_xichen.npz       | canonical+psn   | **xichen torch 2.6.0+cu124 (py3.11)** | A100 | **bit-identical to the `_prefix` run (diffbits=0)** |

## Conclusion

The `_prefix` decade chain was launched with the xichen environment's
interpreter (python 3.11.15, torch 2.6.0+cu124) rather than the yjh
environment (python 3.10.19, torch 2.11.0+cu130) used by every other
experiment in the paper.  Torch 2.6 kernels seed last-bit differences from
the first step; the offline snow-soil feedbacks amplify them (day 10 SWE
max 7e-8 kg m-2 -> year-end rms ~1e-3, skin-temperature max 2.8 K).  The
permanent-snow reset, GPU architecture (A6000 vs A100), the ZCVW guard, and
all local model.py lineage variants are bit-identical controls.

The rerun chain (yjh interpreter, torch 2.11) is therefore the run of
record.  Consequence for manifests: pin `sys.executable`, `torch.__version__`,
and GPU UUID (create_multiyear_manifest.py now records all three plus
per-file working-tree source hashes).
