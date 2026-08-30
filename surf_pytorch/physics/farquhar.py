"""Farquhar 光合作用 + Ball-Berry 气孔导度模型（``FARQUHAR``）。

对耦合的光合作用 / 气孔导度模型（Farquhar et al. 1980; Collatz et al.
1992; Yin et al. 2009 的 C3/C4 三次方程闭合；Ball et al. 1987）的
向量化、可微分 PyTorch 移植。该移植与一个 *独立的* 独立 Fortran 重实现
（``surf_ref.F90::SURF_REF_FARQUHAR``）在 1e-12 相对容差下进行了校验——
两份转写共享同一套 :class:`FarquharConstants` 参数化，因此任何失配都纯粹
是算术层面的，而这正是 ULP 测试旨在暴露的问题。

算法（与 farquhar_mod.F90 + sufarquhar_mod.F90 一致）
---------------------------------------------------------
对每个有植被、白天有短波辐射、LAI 非平凡且生长温度在 ``(-4, 55)``
摄氏度的柱体：

1. 按 ``KVTYPE`` 对每个 PFT 查表获取 Vcmax25 / g0 / a1 / b1 等参数。
2. 对 KmC / KmO / Sco / gamma_star / Rd 采用（修正的）Arrhenius 温度
   标度；对 Jmax / Vcmax 采用 Kattge&Knorr (2007) 驯化方案。
3. Yin (2009) 的 C3 三次方程（附录 B）或 C4 三次方程（附录 C）气孔
   闭合：通过三角恒等式 ``cos(acos(q)/3)``（Baldocchi 1994）求缺项
   三次方程的根，并在两种限制过程（Rubisco 限制与光限制）上迭代两次。
4. 在 ``NLAI=20`` 个 Beer-Lambert 光照层上作冠层积分。
5. Ball-Berry 导度；单位换算到 m/s 和 kgC/m^2/s。

C3/C4 三次方程求根使用 ``torch.acos`` / ``torch.cos``，因此完全可微；
两迭代的 ``LIMIT_PHOTO`` 循环以及对两个根取 ``min`` 的选择按原文逐字
复现，因为上游的循环退出语义很关键（C4 保留 J 限制的值，C3 保留较小
的根）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from surf_pytorch.constants import R as RGAS


# --------------------------------------------------------------------------- #
# Input / output containers
# --------------------------------------------------------------------------- #
@dataclass
class FarquharInputs:
    """Farquhar 光合作用模型所需的逐柱状态。

    所有张量均为 ``(Ngrid,)`` 形状——模型是逐柱的（瓦片维度由调用方
    折叠；上游 FARQUHAR 同样是按每个同化瓦片以其逐瓦片状态调用一次）。
    各字段与 ``FARQUHAR`` 的哑元列表对应（farquhar_mod.F90:73-75）。
    """

    # State.
    ptskm:  torch.Tensor        # tile skin temperature [K]
    ptm:    torch.Tensor        # air temperature at lowest level [K]
    ptsoil: torch.Tensor        # top-soil-layer temperature [K] (growth-T proxy)
    paphm:  torch.Tensor        # air pressure at lowest level [Pa]
    pqm:    torch.Tensor        # air specific humidity at lowest level
    pqsurf: torch.Tensor        # saturation specific humidity at skin T
    pqs:    torch.Tensor        # qsat at skin T (active path; PQSURF kept for fidelity)
    pco2:   torch.Tensor        # CO2 mixing ratio at lowest level [kg/kg]

    # Vegetation descriptors (per grid).
    lai:    torch.Tensor        # leaf-area index [m^2/m^2]
    kvtype: torch.Tensor        # vegetation type (long, 0..20)
    kco2typ: torch.Tensor       # photosynthetic pathway (4=C4, else C3)
    pf2:    torch.Tensor        # soil-moisture stress function [0..1]
    ldland: torch.Tensor        # land mask (bool)

    # Forcing.
    psrfd:  torch.Tensor        # downward SW radiation [W/m^2]


@dataclass
class FarquharOutputs:
    """``FARQUHAR`` 的光合作用 / 导度输出。"""

    pgstot: torch.Tensor        # stomatal conductance [m s^-1]
    pgpp:   torch.Tensor        # gross primary productivity [kgC m^-2 s^-1]
    prd:    torch.Tensor        # dark respiration [kgC m^-2 s^-1]
    pan:    torch.Tensor        # net assimilation [kgC m^-2 s^-1]


# --------------------------------------------------------------------------- #
# Arrhenius (modified) -- eq. of Yin et al. (2009); farquhar_mod.F90:48-67.
# --------------------------------------------------------------------------- #
def _arrhenius_mod(
    ptemp: torch.Tensor, pref: float, peact: torch.Tensor,
    pdeact: torch.Tensor, pentr: torch.Tensor, prr: float,
) -> torch.Tensor:
    """带高温失活项的修正 Arrhenius 函数。

    ``f(T) = exp((T-Tref)*Ea/(Tref*R*T)) * (1+exp((Tref*Entr-Deact)/(Tref*R)))``
    ``                / (1+exp((T*Entr-Deact)/(R*T)))``

    沿 ``ptemp`` 的首维（Ngrid,）向量化。``peact`` / ``pdeact`` /
    ``pentr`` 可以是标量或逐网格张量。
    """
    num = ((ptemp - pref) * peact) / (pref * prr * ptemp)
    denom_num = 1.0 + torch.exp((pref * pentr - pdeact) / (pref * prr))
    denom_den = 1.0 + torch.exp((ptemp * pentr - pdeact) / (prr * ptemp))
    return torch.exp(num) * denom_num / denom_den


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def farquhar(inp: FarquharInputs, veg, consts, debug: dict | None = None) -> FarquharOutputs:
    """向量化、可微分的 Farquhar 光合作用 + 导度。

    Args
    ----
    inp
        逐柱状态（见 :class:`FarquharInputs`）。
    veg
        :class:`~surf_pytorch.tables.veg.VegTable`，携带八张逐 PFT 的
        ``(21, 2)`` Farquhar 表（Vcmax25, g0, a1, b1, ...）以及
        :class:`FarquharConstants` 标量包。
    consts
        :class:`~surf_pytorch.constants.ConstantBundle`（仅用到
        ``pi``；气体常数 ``R`` 来自 ``constants.R``，以与离线驱动
        的 SUCST 严格一致）。

    Returns
    -------
    :class:`FarquharOutputs`，其 ``pgstot`` / ``pgpp`` / ``prd`` / ``pan``
    形状为 ``(Ngrid,)``，dtype/device 与 ``inp.psrfd`` 一致。
    """
    from surf_pytorch.tables.veg import FarquharConstants

    device = inp.psrfd.device
    dtype = inp.psrfd.dtype
    ngrid = inp.psrfd.shape[0]

    fc = veg.farquhar_consts
    if not isinstance(fc, FarquharConstants):
        fc = FarquharConstants()
    nlai = int(fc.nlai)

    # Scalar constants as tensors (so every op stays in dtype/device).
    # Cache per (value, dtype, device) so CUDA-graph capture reuses the same
    # device tensor instead of building a new one (which would require a
    # host->device copy, forbidden during stream capture).
    _cache = farquhar.__dict__.setdefault("_scalar_cache", {})
    def _s(x: float) -> torch.Tensor:
        key = (float(x), dtype, str(device))
        t = _cache.get(key)
        if t is None:
            t = torch.as_tensor(float(x), dtype=dtype, device=device)
            _cache[key] = t
        return t

    def _v(vals, _cache_v=farquhar.__dict__.setdefault("_table_cache", {})):
        key = (tuple(float(v) for v in vals), dtype, str(device))
        t = _cache_v.get(key)
        if t is None:
            t = torch.tensor(list(vals), dtype=dtype, device=device)
            _cache_v[key] = t
        return t

    rr_c   = _s(float(RGAS))
    rpi_c  = consts.pi.to(dtype=dtype, device=device)  # 0-d tensor, avoid .item()
    rmair  = _s(fc.rmair)
    rmco2  = _s(fc.rmco2)
    rtp_00 = _s(fc.rtp_00)
    rpb_std = _s(fc.rpb_std)
    rgb_ref = _s(fc.rgb_ref)
    rtetens_1 = _s(fc.rtetens_1)
    rtetens_2 = _s(fc.rtetens_2)
    rref_temp = _s(fc.rref_temp)
    rmol_to_m_1 = _s(fc.rmol_to_m_1)
    rratio_h2o_to_co2 = _s(fc.rratio_h2o_to_co2)
    rn_vert_att = _s(fc.rn_vert_att)
    rlai_level_depth = _s(fc.rlai_level_depth)
    rlaimax = _s(fc.rlaimax)
    rext_coeff = _s(fc.rext_coeff)
    rrw_to_mol = _s(fc.rw_to_mol)
    rrg_to_par = _s(fc.rrg_to_par)
    ralpha_ll = _s(fc.ralpha_ll)
    rtheta = _s(fc.rtheta)
    rfpsir = _s(fc.rfpsir)
    rfq = _s(fc.rfq)
    rfpseudo = _s(fc.rfpseudo)
    rh_protons = _s(fc.rh_protons)
    rgbs = _s(fc.rgbs)
    roi = _s(fc.roi)
    ralpha = _s(fc.ralpha)
    rkp = _s(fc.rkp)
    re_kmc = _s(fc.re_kmc)
    re_kmo = _s(fc.re_kmo)
    re_rd = _s(fc.re_rd)
    re_sco = _s(fc.re_sco)
    re_gamma_star = _s(fc.re_gamma_star)
    rgamma_star25 = _s(fc.rgamma_star25)
    rstress_gm = _s(fc.rstress_gm)
    rstress_gs = _s(fc.rstress_gs)
    rstress_vcmax = _s(fc.rstress_vcmax)
    re_gamma_star_s = _s(fc.re_gamma_star)
    eps = _s(torch.finfo(dtype).eps)

    # ------------------------------------------------------------------ #
    # Per-PFT table gather -- index by kvtype (0..20).
    # Tables are (21, 2); col 0 = C3, col 1 = C4 (matching the Fortran
    # RVCMAX25(:,1/2) indexing; we'll select the C3 or C4 column per grid
    # after the ctype decision below).
    # ------------------------------------------------------------------ #
    idx = inp.kvtype.long().clamp(min=0).to(device)
    # kvtype_tempo: 0 -> 1 (protection, farquhar_mod.F90:523-527).
    ktv = torch.where(inp.kvtype > 0, inp.kvtype, torch.ones_like(inp.kvtype))
    ktv_i = ktv.long().clamp(min=0, max=20)

    # ctype: 1=C3 (default), 2=C4 when kco2typ==4.
    is_c4 = (inp.kco2typ == 4)
    zctype = torch.where(is_c4, torch.full_like(inp.kvtype, 2, dtype=dtype),
                                 torch.full_like(inp.kvtype, 1, dtype=dtype))
    col_c3 = torch.zeros(ngrid, dtype=torch.long, device=device)   # 0 -> C3 col
    col_c4 = torch.ones(ngrid, dtype=torch.long, device=device)    # 1 -> C4 col
    col = torch.where(is_c4, col_c4, col_c3)                       # (Ngrid,) long

    # Gather each (21,2) table at (ktv_i, col) -> (Ngrid,).
    def _g2(name: str) -> torch.Tensor:
        tbl = getattr(veg, name).to(dtype=dtype, device=device)    # (21, 2)
        row = tbl.index_select(0, ktv_i)                           # (Ngrid, 2)
        return row.gather(1, col.view(-1, 1)).squeeze(1)           # (Ngrid,)

    zvcmax25 = _g2("rvmax25")
    rhumrel  = _g2("rhumrel")
    ra1      = _g2("ra1")
    rb1      = _g2("rb1")
    rg0      = _g2("rg0")
    rgm25    = _g2("rgm25")
    # re_vcmax / re_jmax are per-PFT (vary by veg type); gather the same way.
    re_vcmax = _g2("re_vcmax")
    re_jmax  = _g2("re_jmax")

    # ------------------------------------------------------------------ #
    # Derived scalars per grid.
    # ------------------------------------------------------------------ #
    zpb = inp.paphm / 100.0
    zco2 = inp.pco2 * 1.0e6 * rmair / rmco2

    # VPD (kPa).
    zvpd = ((inp.pqs * zpb / (rtetens_1 + inp.pqs * rtetens_2))
            - (inp.pqm * zpb / (rtetens_1 + inp.pqm * rtetens_2))) / 10.0

    ztemp_growth = inp.ptsoil - rtp_00
    # LSKIN_TEMP=True default: both branches are inp.ptskm, so the where is a
    # no-op; take it directly (also avoids a graph-unsafe host scalar tensor).
    zleaf_temp = inp.ptskm
    # The upstream honours LSKIN_TEMP (default True in offline mode);
    # here we always use skin T.  A caller wanting air-T can swap ptskm.

    # Day respiration Arrhenius (computed for all columns, even nighttime).
    zt_rd = torch.exp(((zleaf_temp - rref_temp) * re_rd)
                      / (rref_temp * rr_c * zleaf_temp))

    # Water-stress limiter.
    zwater_lim = torch.clamp(inp.pf2 * rhumrel, max=1.0)

    # ------------------------------------------------------------------ #
    # LAI level table + Beer-Lambert light profile (nlai+1 boundaries).
    # Upstream ZLAITAB(JLEVEL) for JLEVEL=1..nlai+1 with ZLAITAB(1)=0;
    # ZLIGHT(JLEVEL) = exp(-k*ZLAITAB(JLEVEL)) for JLEVEL=1..nlai so
    # ZLIGHT(1)=1.0.  Our zlaitab[0..nlai] mirrors ZLAITAB(1..nlai+1), so
    # zlight must index zlaitab[0..nlai-1] (NOT [1..nlai]).
    # ------------------------------------------------------------------ #
    jlev_b = torch.arange(0, nlai + 1, dtype=dtype, device=device)   # 0..nlai
    denom_lai = torch.exp(rlai_level_depth * float(nlai)) - 1.0
    zlaitab = rlaimax * (torch.exp(rlai_level_depth * jlev_b) - 1.0) / denom_lai
    # ZLIGHT (nlai,) = exp(-k * ZLAITAB(1..nlai)) = exp(-k * zlaitab[0..nlai-1]).
    zlight = torch.exp(-rext_coeff * zlaitab[:-1])                   # (nlai,)

    # ------------------------------------------------------------------ #
    # Assimilation gate (farquhar_mod.F90:609-612): land AND lai>0.01 AND
    # kvtype>0 AND daytime AND warm enough.
    # ------------------------------------------------------------------ #
    gate_veg = (inp.lai > 0.01) & (inp.kvtype > 0) & inp.ldland
    gate_assim = (inp.psrfd > eps) & (inp.pf2 > eps) \
        & (ztemp_growth > _s(-4.0)) & (ztemp_growth < _s(55.0))
    active = gate_veg & gate_assim                                 # (Ngrid,) bool

    # ------------------------------------------------------------------ #
    # Arrhenius scalings (only meaningful where active, but cheap to compute
    # everywhere and mask later).
    # ------------------------------------------------------------------ #
    zt_kmc = torch.exp(((zleaf_temp - rref_temp) * re_kmc)
                       / (rref_temp * rr_c * zleaf_temp))
    zt_kmo = torch.exp(((zleaf_temp - rref_temp) * re_kmo)
                       / (rref_temp * rr_c * zleaf_temp))
    zt_sco = torch.exp(((zleaf_temp - rref_temp) * re_sco)
                       / (rref_temp * rr_c * zleaf_temp))
    zt_gamma_star = torch.exp(((zleaf_temp - rref_temp) * re_gamma_star_s)
                              / (rref_temp * rr_c * zleaf_temp))

    # Path-pair scalars as (2,) tensors for clean C3/C4 selection.
    rarjv = _v(fc.rarJV)
    rbrjv = _v(fc.rbrJV)
    rasj  = _v(fc.rasJ)
    rbsj  = _v(fc.rbsJ)
    rasv  = _v(fc.rasV)
    rbsv  = _v(fc.rbsV)
    rd_vcmax = _v(fc.rd_vcmax)
    rd_jmax  = _v(fc.rd_jmax)
    re_gm    = _v(fc.re_gm)
    rs_gm    = _v(fc.rs_gm)
    rs_jmax  = _v(fc.rs_jmax)
    rd_gm    = _v(fc.rd_gm)
    rkmc25   = _v(fc.rkmc25)
    rkmo25   = _v(fc.rkmo25)
    rsco25   = _v(fc.rsco25)
    path_idx = (col.to(dtype=dtype))                                # 0=C3, 1=C4

    def _sel(v2: torch.Tensor) -> torch.Tensor:
        return v2.index_select(0, col)                             # (Ngrid,)

    rarjv_g = _sel(rarjv); rbrjv_g = _sel(rbrjv)
    rasj_g  = _sel(rasj);  rbsj_g  = _sel(rbsj)
    rasv_g  = _sel(rasv);  rbsv_g  = _sel(rbsv)
    rd_vcmax_g = _sel(rd_vcmax); rd_jmax_g = _sel(rd_jmax)
    re_gm_g = _sel(re_gm); rs_gm_g = _sel(rs_gm)
    rs_jmax_g = _sel(rs_jmax); rd_gm_g = _sel(rd_gm)
    rkmc25_g = _sel(rkmc25); rkmo25_g = _sel(rkmo25); rsco25_g = _sel(rsco25)

    # Kattge&Knorr acclimation growth-T clamp.
    tg = torch.clamp(ztemp_growth, min=11.0, max=35.0)

    # Jmax / Vcmax at leaf T (modified Arrhenius with acclimation entropy).
    zs_jmax_ac = rasj_g + rbsj_g * tg
    zs_vcmax_ac = rasv_g + rbsv_g * tg
    zt_jmax = _arrhenius_mod(zleaf_temp, float(fc.rref_temp),
                             re_jmax, rd_jmax_g, zs_jmax_ac, float(RGAS))
    zt_vcmax = _arrhenius_mod(zleaf_temp, float(fc.rref_temp),
                              re_vcmax, rd_vcmax_g, zs_vcmax_ac, float(RGAS))

    # C3: ZVJ = ZJMAX25 * ZT_JMAX; ZVC = ZVCMAX25 * ZT_VCMAX.
    # C4: ZVJ = (RARJV+RBRJV*tg)*ZVCMAX25*ZT_JMAX; ZVC = ZVCMAX25*ZT_VCMAX.
    zjmax25_c3 = rarjv_g + rbrjv_g * tg
    zvc_c3 = zvcmax25 * zt_vcmax
    zvj_c3 = zjmax25_c3 * zvcmax25 * zt_jmax
    zvc_c4 = zvcmax25 * zt_vcmax
    zvj_c4 = (rarjv_g + rbrjv_g * tg) * zvcmax25 * zt_jmax
    zvc = torch.where(is_c4, zvc_c4, zvc_c3)
    zvj = torch.where(is_c4, zvj_c4, zvj_c3)

    # Mesophyll conductance (C3 only; C4 branch leaves ZGM=0).
    zt_gm = _arrhenius_mod(zleaf_temp, float(fc.rref_temp),
                           re_gm_g, rd_gm_g, rs_gm_g, float(RGAS))
    zgm_c3 = rgm25 * zt_gm * torch.maximum(1.0 - rstress_gm, zwater_lim.min())
    zgm = torch.where(is_c4, torch.zeros_like(zgm_c3),
                      rgm25 * zt_gm * torch.maximum(1.0 - rstress_gm, zwater_lim))

    zt_gm_c3 = zt_gm   # used below for C3 path
    zgm = torch.where(is_c4, torch.zeros_like(zgm_c3), zgm)

    # g0, KmC/KmO/Sco, FVPD.
    zg0var = rg0 * torch.maximum(1.0 - rstress_gs, zwater_lim)
    zkmc = rkmc25_g * zt_kmc
    zkmo = rkmo25_g * zt_kmo
    zsco = rsco25_g * zt_sco
    inner = torch.clamp(ra1 - rb1 * zvpd, min=eps,
                        max=1.0 - eps)
    zfvpd = (1.0 / (1.0 / inner - 1.0)) * torch.maximum(1.0 - rstress_gs, zwater_lim)

    # Gamma_star and boundary-layer conductances.
    zgamma_star = rgamma_star25 * zt_gamma_star
    zlow_gamma_star = 0.5 / zsco
    zgbh2o = rgb_ref * 44.6 * (rtp_00 / inp.ptm) * (zpb / rpb_std)
    zgbco2 = zgbh2o / rratio_h2o_to_co2

    # ------------------------------------------------------------------ #
    # Canopy integration over the 20 LAI levels.  We compute per-level
    # arrays of shape (Ngrid, nlai) and reduce.  Levels where
    # ZLAITAB(JLEVEL) > PLAI are masked out (the upstream gather's active
    # set is a level prefix since ZLAITAB is monotonic).
    # ------------------------------------------------------------------ #
    plai = inp.lai                                                       # (Ngrid,)
    # Active level mask: upstream ZLAITAB(JLEVEL) <= PLAI for JLEVEL=1..nlai.
    # Since zlaitab[0]=ZLAITAB(1)=0, this is zlaitab[0..nlai-1] <= plai.
    lev_lai = zlaitab[:-1].unsqueeze(1)                                     # (nlai, 1)
    level_active = (lev_lai <= plai.unsqueeze(0)) & active.unsqueeze(0)     # (nlai, Ngrid)
    # Per-level LAI slab thickness ZLAITAB(JLEVEL+1)-ZLAITAB(JLEVEL).
    dlai = (zlaitab[1:] - zlaitab[:-1]).unsqueeze(1)                        # (nlai, 1)

    # Nitrogen attenuation (nlai,) -> (nlai, 1) for broadcast with (Ngrid,).
    znitro = (1.0 - rn_vert_att * (1.0 - zlight)).unsqueeze(1)            # (nlai, 1)
    # Per-level Rd (nlai, Ngrid).  ZNITRO_VCMAX varies by level; the rest
    # varies by grid.  Broadcast: (nlai,1) * (1,Ngrid).
    stress_vc_col = torch.maximum(1.0 - rstress_vcmax, zwater_lim)        # (Ngrid,)
    zrd_lev = (znitro * 0.01
               * (zvcmax25 * zt_rd * stress_vc_col).unsqueeze(0))        # (nlai, Ngrid)

    # Vc2/J2 scaled into the canopy.  (nlai,1) * (1,Ngrid) -> (nlai, Ngrid).
    zvc2 = znitro * (zvc * stress_vc_col).unsqueeze(0)
    zvj2 = znitro * (zvj * stress_vc_col).unsqueeze(0)

    # Eq. 4 of Yin (2009): electron transport rate J.
    ziabs = (inp.psrfd.unsqueeze(0) * rrw_to_mol * rrg_to_par * rext_coeff
             * zlight.unsqueeze(1))                                     # (nlai, Ngrid)
    zjmax_lev = zvj2
    zjj = ((ralpha_ll * ziabs + zjmax_lev
            - torch.sqrt((ralpha_ll * ziabs + zjmax_lev).pow(2)
                         - 4.0 * rtheta * zjmax_lev * ralpha_ll * ziabs))
           / (2.0 * rtheta))                                            # (nlai, Ngrid)

    # ------------------------------------------------------------------ #
    # C3 / C4 cubic closures (vectorised over both levels and grids).
    # ------------------------------------------------------------------ #
    # Common helpers expanded to (nlai, Ngrid).
    zco2_L  = zco2.unsqueeze(0)
    zg0var_L = zg0var.unsqueeze(0)
    zfvpd_L = zfvpd.unsqueeze(0)
    zgbco2_L = zgbco2.unsqueeze(0)
    zgamma_star_L = zgamma_star.unsqueeze(0)
    zlow_gamma_star_L = zlow_gamma_star.unsqueeze(0)
    zkmc_L = zkmc.unsqueeze(0)
    zkmo_L = zkmo.unsqueeze(0)
    zsco_L = zsco.unsqueeze(0)
    zgm_L  = zgm.unsqueeze(0)
    is_c4_L = is_c4.unsqueeze(0)

    # ---- C4 cubic (Yin 2009 Appendix C) --------------------------------
    zfcyc = (1.0 - (4.0 * (1.0 - rfpsir) * (1.0 + rfq) + 3.0 * rh_protons * rfpseudo)
             / (3.0 * rh_protons - 4.0 * (1.0 - rfpsir)))
    zrm = zrd_lev / 2.0
    zcs_star = ((rgbs * zlow_gamma_star_L * roi
                 - (1.0 + zlow_gamma_star_L * ralpha / 0.047) * zrd_lev
                 + zrm) / (rgbs + rkp))
    zj2 = zjj / (1.0 - rfpseudo / (1.0 - zfcyc))
    zzc = (2.0 + rfq - zfcyc) / (rh_protons * (1.0 - zfcyc))
    zvpj2 = rfpsir * zj2 * zzc / 2.0

    # Two-iteration LIMIT_PHOTO loop for C4.  Upstream keeps the LAST
    # iteration's ZGS (loop-exit value = J-limited).  We compute both
    # roots explicitly and select iter-2's value where the cubic is valid.
    za_3 = torch.full_like(zjj, 9999.0)
    zgsc4 = torch.zeros_like(zjj)
    za_c4_final = None
    zgsc4_final = None
    # Vc-limited re-init values for the C4 ELSE branch (upstream 958-965).
    za_vc = 1.0 + rkp / rgbs
    zx1_vc = zvc2
    zx2_vc = zkmc_L / zkmo_L
    zx3_vc = zkmc_L
    for limit_photo in (1, 2):
        if limit_photo == 1:
            za_i = 1.0 + rkp / rgbs
            zb_i = torch.zeros_like(za_i)
            zx1 = zvc2
            zx2 = zkmc_L / zkmo_L
            zx3 = zkmc_L
        else:
            za_i = torch.ones_like(zcs_star)
            zb_i = zvpj2
            zx1 = (1.0 - rfpsir) * zj2 * zzc / 3.0
            zx2 = 7.0 * zlow_gamma_star_L / 3.0
            zx3 = torch.zeros_like(zx1)
        zm = zfvpd_L - zg0var_L / zgbco2_L
        zd = (zg0var_L * (zco2_L - zcs_star) + zfvpd_L * zrd_lev)
        zf = ((zb_i - zrm - zlow_gamma_star_L * roi * rgbs) * zx1 * zd
              + za_i * rgbs * zx1 * zco2_L * zd)
        zj_c = (((zb_i - zrm + rgbs * zx3 + zx2 * rgbs * roi) * zm)
                + ((ralpha * zx2 / 0.047 - 1.0) * zd)
                + (za_i * rgbs * (zco2_L * zm - zd / zgbco2_L - (zco2_L - zcs_star))))
        zg = (((zb_i - zrm - zlow_gamma_star_L * roi * rgbs) * zx1 * zm)
              - ((ralpha * zlow_gamma_star_L / 0.047 + 1.0) * zx1 * zd)
              + (za_i * rgbs * zx1
                 * (zco2_L * zm - zd / zgbco2_L - (zco2_L - zcs_star))))
        zh = -(((ralpha * zlow_gamma_star_L / 0.047 + 1.0) * zx1 * zm)
               + (za_i * rgbs * zx1 * (zm - 1.0)) / zgbco2_L)
        zi_c = ((zb_i - zrm + rgbs * zx3 + zx2 * rgbs * roi) * zd
                + za_i * rgbs * zco2_L * zd)
        zl_c = ((ralpha * zx2 / 0.047 - 1.0) * zm
                - (za_i * rgbs * (zm - 1.0)) / zgbco2_L)
        zp_c = (zj_c - (zh - zl_c * zrd_lev)) / zl_c
        zq_c = (zi_c + zj_c * zrd_lev - zg) / zl_c
        zr_c = -(zf - zi_c * zrd_lev) / zl_c

        zqq = (zp_c.pow(2) - 3.0 * zq_c) / 9.0
        zuu = (2.0 * zp_c.pow(3) - 9.0 * zp_c * zq_c + 27.0 * zr_c) / 54.0
        valid = (zqq >= 0.0) & (torch.abs(zuu / (torch.abs(zqq).pow(1.5) + eps)) <= 1.0)
        zpsi = torch.acos(torch.clamp(zuu / (zqq.pow(1.5) + eps), min=-1.0, max=1.0))
        za_3_tmp = (-2.0 * torch.sqrt(torch.clamp_min(zqq, min=0.0))
                    * torch.cos((zpsi + 4.0 * rpi_c) / 3.0) - zp_c / 3.0)

        # Update za_3 with the minimum; on the ELSE branch re-init
        # ZA/ZB/ZX1/ZX2/ZX3 for Vc-limited (upstream 958-965) so the
        # post-block ZGS closure sees the Vc path.
        take_min = valid & (za_3_tmp < za_3)
        za_3 = torch.where(take_min, za_3_tmp, za_3)
        else_reinit = valid & ~take_min
        za_i = torch.where(else_reinit, za_vc, za_i)
        zb_i = torch.where(else_reinit, torch.zeros_like(zb_i), zb_i)
        zx1 = torch.where(else_reinit, zx1_vc, zx1)
        zx2 = torch.where(else_reinit, zx2_vc, zx2)
        zx3 = torch.where(else_reinit, zx3_vc, zx3)

        # ZGS closure (computed every iteration in upstream C4; the loop-exit
        # value is iter-2's, which we keep).
        za_3_use = torch.where((za_3 == 9999.0) | (za_3 < (-zrd_lev)),
                               -zrd_lev, za_3)
        zassimi_c4 = za_3_use
        near_zero = (torch.abs(zassimi_c4 + zrd_lev) < eps)
        zobs = (ralpha * zassimi_c4) / (0.047 * rgbs) + roi
        zcc_c4 = (((zassimi_c4 + zrd_lev) * (zx2 * zobs + zx3)
                   + zlow_gamma_star_L * zobs * zx1)
                  / torch.clamp_min(zx1 - (zassimi_c4 + zrd_lev), min=eps))
        zgs_c4 = (zg0var_L + (zassimi_c4 + zrd_lev) / (zco2_L - zcs_star) * zfvpd_L)
        zgs_iter = torch.where(near_zero, zg0var_L, zgs_c4)
        # Loop-exit semantics: iter-2 wins.
        if limit_photo == 2:
            zgsc4_final = zgs_iter
            zassimi_c4_final = zassimi_c4

    # ---- C3 cubic (Yin 2009 Appendix B) --------------------------------
    # Vc-limited x1/x2 (re-used in both the LIMIT_PHOTO=1 init and the
    # ELSE-branch re-initialisation when the new root does not improve).
    zx1c_vc = zvc2
    zx2c_vc = zkmc_L * (1.0 + 2.0 * zgamma_star_L * zsco_L / zkmo_L)
    za_1 = torch.full_like(zjj, 9999.0)
    zx1c = zx1c_vc.clone()
    zx2c = zx2c_vc.clone()
    for limit_photo in (1, 2):
        if limit_photo == 1:
            zx1c = zx1c_vc.clone()
            zx2c = zx2c_vc.clone()
        else:
            zx1c = zjj / 4.0
            zx2c = 2.0 * zgamma_star_L
        za_c = (zg0var_L * (zx2c + zgamma_star_L)
                + (zg0var_L / zgm_L + zfvpd_L) * (zx1c - zrd_lev))
        zb_c = (zco2_L * (zx1c - zrd_lev) - zgamma_star_L * zx1c - zrd_lev * zx2c)
        zc_c = (zco2_L + zx2c + (1.0 / zgm_L + 1.0 / zgbco2_L) * (zx1c - zrd_lev))
        zd_c = (zx2c + zgamma_star_L + (zx1c - zrd_lev) / zgm_L)
        zm_c = (1.0 / zgm_L + (zg0var_L / zgm_L + zfvpd_L)
                * (1.0 / zgm_L + 1.0 / zgbco2_L))
        zp_c = -(zd_c + (zx1c - zrd_lev) / zgm_L + za_c
                 * (1.0 / zgm_L + 1.0 / zgbco2_L)
                 + (zg0var_L / zgm_L + zfvpd_L) * zc_c) / zm_c
        zq_c = (zd_c * (zx1c - zrd_lev) + za_c * zc_c
                + (zg0var_L / zgm_L + zfvpd_L) * zb_c) / zm_c
        zr_c = -za_c * zb_c / zm_c
        zqq = (zp_c.pow(2) - 3.0 * zq_c) / 9.0
        zuu = (2.0 * zp_c.pow(3) - 9.0 * zp_c * zq_c + 27.0 * zr_c) / 54.0
        valid = (zqq >= 0.0) & (torch.abs(zuu / (torch.abs(zqq).pow(1.5) + eps)) <= 1.0)
        zpsi = torch.acos(torch.clamp(zuu / (zqq.pow(1.5) + eps), min=-1.0, max=1.0))
        za_1_tmp = (-2.0 * torch.sqrt(torch.clamp_min(zqq, min=0.0))
                    * torch.cos(zpsi / 3.0) - zp_c / 3.0)
        take_min = valid & (za_1_tmp < za_1)
        za_1 = torch.where(take_min, za_1_tmp, za_1)
        # ELSE-branch re-init: when the candidate root does NOT improve the
        # running minimum, re-initialise x1/x2 to the Vc-limited values so
        # the post-loop ZGS closure (which reuses ZX1/ZX2) sees the Vc path.
        else_reinit = valid & ~take_min
        zx1c = torch.where(else_reinit, zx1c_vc, zx1c)
        zx2c = torch.where(else_reinit, zx2c_vc, zx2c)
    za_1_use = torch.where((za_1 == 9999.0) | (za_1 < (-zrd_lev)),
                           -zrd_lev, za_1)
    zassimi_c3 = za_1_use
    near_zero_c3 = (torch.abs(zassimi_c3 + zrd_lev) < eps)
    zcc_c3 = ((zgamma_star_L * zx1c + (zassimi_c3 + zrd_lev) * zx2c)
              / torch.clamp_min(zx1c - (zassimi_c3 + zrd_lev), min=eps))
    zleaf_ci = zcc_c3 + zassimi_c3 / zgm_L
    zci_star = zgamma_star_L - zrd_lev / zgm_L
    zgs_c3 = (zg0var_L + (zassimi_c3 + zrd_lev) / (zleaf_ci - zci_star) * zfvpd_L)
    zgsc3_final = torch.where(near_zero_c3, zg0var_L, zgs_c3)

    # Select C3 vs C4 assimilation / conductance per (level, grid).
    zassimi_lev = torch.where(is_c4_L, zassimi_c4_final, zassimi_c3)
    zgs_lev = torch.where(is_c4_L, zgsc4_final, zgsc3_final)

    if debug is not None:
        debug.update(dict(
            zvcmax25=zvcmax25, zt_vcmax=zt_vcmax, zt_jmax=zt_jmax, zvc=zvc, zvj=zvj,
            zgm=zgm, zg0var=zg0var, zkmc=zkmc, zkmo=zkmo, zsco=zsco,
            zgamma_star=zgamma_star, zfvpd=zfvpd, zgbco2=zgbco2, zvpd=zvpd, zco2=zco2,
            zrd_lev=zrd_lev, zvc2=zvc2, zvj2=zvj2, zjj=zjj,
            zassimi_c3=zassimi_c3, zgsc3_final=zgsc3_final,
            zassimi_c4_final=zassimi_c4_final, zgsc4_final=zgsc4_final,
            zassimi_lev=zassimi_lev, zgs_lev=zgs_lev, level_active=level_active,
            dlai=dlai.squeeze(1), zlight=zlight,
        ))

    # ------------------------------------------------------------------ #
    # Canopy sums.  Top-level conductance = ZGS at level 1 (per grid).
    # Assimilation integral: sum (A+Rd)*dlai over active levels.
    # PGSTOT: sum ZGS*dlai over active levels.
    # ------------------------------------------------------------------ #
    contrib = (zassimi_lev + zrd_lev) * dlai
    zassimtot = torch.where(level_active, contrib, torch.zeros_like(contrib)).sum(0)
    gstot_raw = torch.where(level_active, zgs_lev * dlai,
                            torch.zeros_like(zgs_lev)).sum(0)
    # Top-level gs: ZGS at the FIRST active level (JLEVEL==1 in upstream;
    # since ZLAITAB(1) is the smallest, level 1 is active iff any is).
    zleaf_gs_top = zgs_lev[0] * level_active[0].to(dtype=dtype)

    # ZILAI = index of the last active level (upstream NINT(ZILAI)).
    # We need ZLAITAB(NINT(ZILAI)+1) for the ZGSTOP unit conversion.
    any_active = level_active.any(0)
    # last active level index per grid (0-based among 0..nlai-1).
    last_idx = (level_active.to(dtype=torch.long)
                * torch.arange(nlai, device=device).unsqueeze(1)).argmax(0)
    zilai = last_idx.to(dtype=dtype)
    zlaitab_zilai_p1 = zlaitab[zilai.long() + 1]

    # Dark-respiration integral over ALL nlai levels (Rd needed at night).
    zrdtot = (zrd_lev * dlai).sum(0)

    # ------------------------------------------------------------------ #
    # Unit conversions (farquhar_mod.F90:1216,1237-1242,1305-1306).
    # ------------------------------------------------------------------ #
    conv_factor = rmol_to_m_1 * (inp.ptm / rtp_00) * (rpb_std / zpb) \
        * rratio_h2o_to_co2
    zgstop = conv_factor * zleaf_gs_top * zlaitab_zilai_p1
    pgstot_raw = conv_factor * gstot_raw

    assimilated = any_active & (zleaf_gs_top > 0.0) | (zassimtot != 0.0)
    pgpp = zassimtot * 1.0e-6 * rmco2
    pgstot = pgstot_raw
    pan = (zassimtot - zrdtot) * 1.0e-6 * rmco2
    prd = zrdtot * 1.0e-6 * rmco2

    # Zero out non-land / non-assimilating columns.
    zero = torch.zeros(ngrid, dtype=dtype, device=device)
    pgstot = torch.where(assimilated, pgstot, zero)
    pgpp = torch.where(assimilated, pgpp, zero)
    pan = torch.where(assimilated, pan, zero)
    prd = torch.where(assimilated, prd, zero)

    return FarquharOutputs(pgstot=pgstot, pgpp=pgpp, prd=prd, pan=pan)


__all__ = ["FarquharInputs", "FarquharOutputs", "farquhar"]
