"""Daily battery MILP (scipy/HiGHS) with the OR-Tools (SCIP) cross-check (PLAN.md §5).

For one delivery day with n hours (23, 24 or 25) and a price vector p (EUR/MWh):

    maximise   sum_h p_h (d_h - c_h) - DEG * sum_h d_h
    subject to 0 <= c_h <= P u_h,  0 <= d_h <= P (1 - u_h),  u_h binary
               soc_{h+1} = soc_h + eta_c c_h - d_h / eta_d,  0 <= soc <= E
               soc_0 = soc_n = SOC_START,  sum_h d_h <= MAX_DISCHARGE

c_h and d_h are the energy bought and sold in hour h (MWh at the grid side; with 1-hour
products this equals the average power in MW). The binary u_h forbids charging and
discharging in the same hour: with negative prices a plain LP would do both to burn energy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from bessrank import config

# Solvers must agree within this relative tolerance (CLAUDE.md rule 7). The floor of 1 EUR
# keeps the check meaningful on days whose optimal profit is close to 0.
AGREEMENT_RTOL = 1e-6


class SolverMismatchError(RuntimeError):
    """HiGHS and SCIP returned different objective values for the same day."""


@dataclass(frozen=True)
class Battery:
    power_mw: float = config.POWER_MW
    capacity_mwh: float = config.CAPACITY_MWH
    eta_charge: float = config.ETA_CHARGE
    eta_discharge: float = config.ETA_DISCHARGE
    soc_start_mwh: float = config.SOC_START_MWH
    max_discharge_mwh: float = config.MAX_DISCHARGE_MWH_PER_DAY
    degradation_eur_per_mwh: float = config.DEGRADATION_EUR_PER_MWH


DEFAULT_BATTERY = Battery()


@dataclass
class Schedule:
    charge: np.ndarray  # MWh bought per hour
    discharge: np.ndarray  # MWh sold per hour
    soc: np.ndarray  # MWh at the end of each hour (n values)
    objective: float  # EUR, at the prices the schedule was optimised on


def solve_day_scipy(prices, battery=DEFAULT_BATTERY):
    """Optimise one day with scipy.optimize.milp (HiGHS). Primary solver."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    if np.isnan(p).any():
        raise ValueError("price vector contains NaN")
    b = battery
    # Variable layout: c_0..c_{n-1}, d_0..d_{n-1}, u_0..u_{n-1}, soc_1..soc_n
    ic, idis, iu, isoc = 0, n, 2 * n, 3 * n
    nvar = 4 * n

    # milp minimises, so the objective is the negative profit.
    cost = np.zeros(nvar)
    cost[ic:ic + n] = p
    cost[idis:idis + n] = -(p - b.degradation_eur_per_mwh)

    A = lil_matrix((3 * n + 1, nvar))
    lo = np.zeros(3 * n + 1)
    hi = np.zeros(3 * n + 1)
    for h in range(n):
        # c_h - P u_h <= 0
        A[h, ic + h] = 1.0
        A[h, iu + h] = -b.power_mw
        lo[h], hi[h] = -np.inf, 0.0
        # d_h + P u_h <= P
        A[n + h, idis + h] = 1.0
        A[n + h, iu + h] = b.power_mw
        lo[n + h], hi[n + h] = -np.inf, b.power_mw
        # soc_{h+1} - soc_h - eta_c c_h + d_h / eta_d = 0, with soc_0 a constant
        r = 2 * n + h
        A[r, isoc + h] = 1.0
        if h > 0:
            A[r, isoc + h - 1] = -1.0
        A[r, ic + h] = -b.eta_charge
        A[r, idis + h] = 1.0 / b.eta_discharge
        lo[r] = hi[r] = b.soc_start_mwh if h == 0 else 0.0
    # at most one equivalent full cycle per day
    A[3 * n, idis:idis + n] = 1.0
    lo[3 * n], hi[3 * n] = -np.inf, b.max_discharge_mwh

    lb = np.zeros(nvar)
    ub = np.concatenate([np.full(2 * n, b.power_mw), np.ones(n), np.full(n, b.capacity_mwh)])
    lb[isoc + n - 1] = ub[isoc + n - 1] = b.soc_start_mwh  # end the day where it started
    integrality = np.zeros(nvar)
    integrality[iu:iu + n] = 1

    # mip_rel_gap=0: solve to proven optimality, otherwise HiGHS stops within 0.01%.
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), bounds=Bounds(lb, ub),
               integrality=integrality, options={"mip_rel_gap": 0.0})
    if not res.success:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    x = res.x
    return Schedule(charge=x[ic:ic + n], discharge=x[idis:idis + n], soc=x[isoc:isoc + n],
                    objective=float(-res.fun))


def solve_day_ortools(prices, battery=DEFAULT_BATTERY):
    """The same program in OR-Tools with SCIP. Used as a cross-check on every day solved."""
    from ortools.linear_solver import pywraplp

    p = np.asarray(prices, dtype=float)
    n = len(p)
    b = battery
    solver = pywraplp.Solver.CreateSolver("SCIP")
    c = [solver.NumVar(0, b.power_mw, f"c{h}") for h in range(n)]
    d = [solver.NumVar(0, b.power_mw, f"d{h}") for h in range(n)]
    u = [solver.BoolVar(f"u{h}") for h in range(n)]
    soc = [solver.NumVar(0, b.capacity_mwh, f"soc{h + 1}") for h in range(n)]
    for h in range(n):
        solver.Add(c[h] <= b.power_mw * u[h])
        solver.Add(d[h] <= b.power_mw * (1 - u[h]))
        previous = b.soc_start_mwh if h == 0 else soc[h - 1]
        solver.Add(soc[h] == previous + b.eta_charge * c[h] - d[h] * (1.0 / b.eta_discharge))
    solver.Add(soc[n - 1] == b.soc_start_mwh)
    solver.Add(solver.Sum(d) <= b.max_discharge_mwh)
    solver.Maximize(solver.Sum([float(p[h]) * (d[h] - c[h]) - b.degradation_eur_per_mwh * d[h]
                                for h in range(n)]))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.0)  # proven optimum, like HiGHS above
    status = solver.Solve(params)
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"SCIP status {status}")
    return float(solver.Objective().Value())


def objectives_agree(a, b, rtol=AGREEMENT_RTOL):
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def solve_day(prices, battery=DEFAULT_BATTERY, cross_check=True):
    """Solve with HiGHS; if cross_check, re-solve with SCIP and stop the run on disagreement."""
    schedule = solve_day_scipy(prices, battery)
    if cross_check:
        other = solve_day_ortools(prices, battery)
        if not objectives_agree(schedule.objective, other):
            raise SolverMismatchError(
                f"HiGHS {schedule.objective!r} vs SCIP {other!r} EUR (rtol {AGREEMENT_RTOL})")
    return schedule


def settle(schedule, actual_prices, battery=DEFAULT_BATTERY):
    """Profit (EUR) of a schedule at the actual prices: energy sold minus bought, minus
    degradation. Price-taker, no fees or grid charges (PLAN.md §5)."""
    p = np.asarray(actual_prices, dtype=float)
    return float(p @ (schedule.discharge - schedule.charge)
                 - battery.degradation_eur_per_mwh * schedule.discharge.sum())


# --- Risk-aware schedule (PLAN.md §8 stretch, exploratory) -----------------------------------
def solve_day_cvar_ortools(scenarios, lam, battery=DEFAULT_BATTERY, beta=0.95, backend="SCIP"):
    """One day's schedule from price scenarios (S x n, EUR/MWh): maximise the mean scenario
    profit minus lam x CVaR_beta of the loss (loss = -profit), with the Rockafellar-Uryasev
    formulation: CVaR = eta + sum_s z_s / ((1 - beta) S), z_s >= loss_s - eta, z_s >= 0.

    Same battery constraints as the daily program above. lam = 0 is the deterministic program
    on the mean scenario price. Returns (Schedule, objective in EUR).
    """
    from ortools.linear_solver import pywraplp

    P = np.asarray(scenarios, dtype=float)
    S, n = P.shape
    b = battery
    solver = pywraplp.Solver.CreateSolver(backend)
    solver.SuppressOutput()
    if backend == "HIGHS":
        # OR-Tools does not pass RELATIVE_MIP_GAP on to HiGHS, which then stops inside its
        # default 1e-4 gap (S4: 1.8e-5 relative below SCIP on a test day). HiGHS's own option
        # applies it, although OR-Tools returns False for the call.
        solver.SetSolverSpecificParametersAsString("mip_rel_gap=0")
    c = [solver.NumVar(0, b.power_mw, f"c{h}") for h in range(n)]
    d = [solver.NumVar(0, b.power_mw, f"d{h}") for h in range(n)]
    u = [solver.BoolVar(f"u{h}") for h in range(n)]
    soc = [solver.NumVar(0, b.capacity_mwh, f"soc{h + 1}") for h in range(n)]
    eta = solver.NumVar(-solver.infinity(), solver.infinity(), "eta")
    z = [solver.NumVar(0, solver.infinity(), f"z{s}") for s in range(S)]
    for h in range(n):
        solver.Add(c[h] <= b.power_mw * u[h])
        solver.Add(d[h] <= b.power_mw * (1 - u[h]))
        previous = b.soc_start_mwh if h == 0 else soc[h - 1]
        solver.Add(soc[h] == previous + b.eta_charge * c[h] - d[h] * (1.0 / b.eta_discharge))
    solver.Add(soc[n - 1] == b.soc_start_mwh)
    solver.Add(solver.Sum(d) <= b.max_discharge_mwh)
    for s in range(S):
        # z_s >= loss_s - eta, with loss_s = -(sum_h p_sh (d_h - c_h) - DEG sum_h d_h)
        ct = solver.Constraint(0.0, solver.infinity())
        ct.SetCoefficient(z[s], 1.0)
        ct.SetCoefficient(eta, 1.0)
        for h in range(n):
            ct.SetCoefficient(d[h], float(P[s, h]) - b.degradation_eur_per_mwh)
            ct.SetCoefficient(c[h], -float(P[s, h]))
    mean_p = P.mean(axis=0)
    objective = solver.Objective()
    for h in range(n):
        objective.SetCoefficient(d[h], float(mean_p[h]) - b.degradation_eur_per_mwh)
        objective.SetCoefficient(c[h], -float(mean_p[h]))
    objective.SetCoefficient(eta, -lam)
    for s in range(S):
        objective.SetCoefficient(z[s], -lam / ((1 - beta) * S))
    objective.SetMaximization()
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, 0.0)
    status = solver.Solve(params)
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"{backend} status {status}")
    schedule = Schedule(charge=np.array([v.solution_value() for v in c]),
                        discharge=np.array([v.solution_value() for v in d]),
                        soc=np.array([v.solution_value() for v in soc]), objective=objective.Value())
    return schedule, objective.Value()


def solve_day_cvar(scenarios, lam, battery=DEFAULT_BATTERY, beta=0.95):
    """CVaR program with SCIP, re-solved with HiGHS; stops the run if the objectives disagree."""
    schedule, obj = solve_day_cvar_ortools(scenarios, lam, battery, beta, "SCIP")
    _, other = solve_day_cvar_ortools(scenarios, lam, battery, beta, "HIGHS")
    if not objectives_agree(obj, other):
        raise SolverMismatchError(f"CVaR lam={lam}: SCIP {obj!r} vs HiGHS {other!r} EUR")
    return schedule
