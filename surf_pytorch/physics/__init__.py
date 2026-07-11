"""Physics sub-modules of the OpenIFS land-surface package.

Modules
-------
* :mod:`surf_pytorch.physics.vlamsk`    -- tiled skin layer conductivity
  (``VLAMSK`` upstream of SURFSEB).
* :mod:`surf_pytorch.physics.seb`       -- surface energy balance /
  Penman-Monteith skin-temperature elimination (``SURFSEB_CTL``).
* :mod:`surf_pytorch.physics.exchange`  -- Monin-Obukhov surface-layer
  exchange coefficients (``VEXCS``, Beljaars 1990).
* :mod:`surf_pytorch.physics.stomata`   -- Jarvis stomatal resistance
  and VEVAP alpha closure (``vsurf_mod`` / ``vevap_mod``).
* :mod:`surf_pytorch.physics.diffusion` -- implicit vertical-diffusion
  SURFSEB wrapper (``VDFDIFH1S`` offline single-column driver).
* :mod:`surf_pytorch.physics.soil`      -- HTESSEL soil temperature
  update (apparent heat capacity + implicit Thomas, ``SRFRCGS`` /
  ``SRFTS`` / ``SRFWDIFS``).
"""

from . import vlamsk
from . import seb
from . import exchange
from . import stomata
from . import diffusion
from . import soil

__all__ = ["vlamsk", "seb", "exchange", "stomata", "diffusion", "soil"]
