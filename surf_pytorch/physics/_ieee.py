"""IEEE-754 helpers for bitwise parity with the Fortran.

torch lowers ``scalar / tensor`` and ``tensor / scalar`` to a
reciprocal-multiply fast path; for non-power-of-two scalars the result
differs from the Fortran's true division in the last bit.  Routing the
division through two tensors keeps the div kernel's single correctly
rounded operation, matching ``A/B`` in the Fortran.
"""

import torch


def true_div(a, b):
    # The scalar operand is materialised as a tensor so the division is
    # one correctly rounded op (no reciprocal-multiply lowering).
    if isinstance(a, (int, float)):
        a = torch.full_like(b, a)
    elif isinstance(b, (int, float)):
        b = torch.full_like(a, b)
    return a / b
