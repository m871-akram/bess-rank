"""Battery MILP: toy days with known answers, DST days, solver agreement, settlement."""
import numpy as np
import pytest

from bessrank import battery, evaluate

B = battery.DEFAULT_BATTERY
ETA = B.eta_charge  # = eta_discharge = sqrt(0.88)


def test_two_price_day_has_the_hand_computed_profit():
    # Cheap at 0 EUR/MWh for 12 hours, then 100 EUR/MWh. The battery starts at 1 MWh and must
    # end at 1 MWh, so the expensive block can only use what was stored above 1 MWh: fill to
    # 2 MWh at 0 EUR (buy 1/eta MWh), then release 1 MWh of storage (sell eta MWh) at 100 EUR.
    prices = np.r_[np.zeros(12), np.full(12, 100.0)]
    s = battery.solve_day(prices)
    assert s.charge.sum() == pytest.approx(1.0 / ETA)
    assert s.discharge.sum() == pytest.approx(ETA)
    assert s.objective == pytest.approx((100.0 - B.degradation_eur_per_mwh) * ETA)
    assert s.soc.max() == pytest.approx(B.capacity_mwh)
    assert s.soc[-1] == pytest.approx(B.soc_start_mwh)


def test_discharge_cap_binds_on_a_double_peak_day():
    # Two cheap/expensive cycles: each could sell eta * 2 MWh, but the day is capped at 2 MWh.
    prices = np.r_[np.zeros(6), np.full(6, 200.0), np.zeros(6), np.full(6, 200.0)]
    s = battery.solve_day(prices)
    assert s.discharge.sum() == pytest.approx(B.max_discharge_mwh)
    assert s.objective == pytest.approx((200.0 - B.degradation_eur_per_mwh) * B.max_discharge_mwh)


def test_flat_prices_do_nothing():
    s = battery.solve_day(np.full(24, 50.0))
    assert s.objective == pytest.approx(0.0, abs=1e-9)
    assert s.discharge.sum() == pytest.approx(0.0, abs=1e-9)


def test_negative_prices_never_charge_and_discharge_at_once():
    # Flat -50: buying c earns 50c, selling back 0.88c costs (50 + 10) * 0.88c = 52.8c. No trade.
    s = battery.solve_day(np.full(24, -50.0))
    assert np.all(np.minimum(s.charge, s.discharge) <= 1e-9)
    assert s.objective == pytest.approx(0.0, abs=1e-9)

    # Flat -80: earning 80c beats paying (80 + 10) * 0.88c = 79.2c, so cycle up to the 2 MWh cap.
    s = battery.solve_day(np.full(24, -80.0))
    assert np.all(np.minimum(s.charge, s.discharge) <= 1e-9)
    assert s.discharge.sum() == pytest.approx(2.0)
    assert s.objective == pytest.approx(80.0 * 2.0 / ETA ** 2 - 90.0 * 2.0)

    prices = np.r_[np.full(6, -100.0), np.full(18, 20.0)]
    s = battery.solve_day(prices)
    assert np.all(np.minimum(s.charge, s.discharge) <= 1e-9)
    assert s.objective > 0  # being paid to charge, then selling


def test_spread_below_losses_is_not_traded():
    # A 5 EUR spread does not cover 12% losses plus 10 EUR/MWh degradation.
    s = battery.solve_day(np.r_[np.full(12, 50.0), np.full(12, 55.0)])
    assert s.objective == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", [23, 24, 25])
def test_dst_day_lengths(n):
    rng = np.random.default_rng(n)
    prices = 80 + 50 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.normal(0, 15, n)
    s = battery.solve_day(prices)  # includes the SCIP cross-check
    assert len(s.charge) == len(s.discharge) == len(s.soc) == n
    assert s.soc[-1] == pytest.approx(B.soc_start_mwh)
    assert np.all(s.soc >= -1e-9) and np.all(s.soc <= B.capacity_mwh + 1e-9)
    assert np.all(s.charge <= B.power_mw + 1e-9) and np.all(s.discharge <= B.power_mw + 1e-9)
    assert s.discharge.sum() <= B.max_discharge_mwh + 1e-9
    soc = B.soc_start_mwh + np.cumsum(B.eta_charge * s.charge - s.discharge / B.eta_discharge)
    np.testing.assert_allclose(s.soc, soc, atol=1e-7)
    assert battery.settle(s, prices) == pytest.approx(s.objective)


def test_solvers_agree_on_random_days():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.choice([23, 24, 25])
        prices = rng.normal(60, 60, n)  # includes negative prices
        a = battery.solve_day_scipy(prices).objective
        b = battery.solve_day_ortools(prices)
        assert battery.objectives_agree(a, b)


def test_mismatch_stops_the_run(monkeypatch):
    monkeypatch.setattr(battery, "solve_day_ortools", lambda prices, bat=None: 1e6)
    with pytest.raises(battery.SolverMismatchError):
        battery.solve_day(np.r_[np.zeros(12), np.full(12, 100.0)])


def test_settle_at_actual_prices():
    s = battery.solve_day(np.r_[np.zeros(12), np.full(12, 100.0)])
    actual = np.r_[np.full(12, 10.0), np.full(12, 50.0)]
    expected = actual @ (s.discharge - s.charge) - B.degradation_eur_per_mwh * s.discharge.sum()
    assert battery.settle(s, actual) == pytest.approx(expected)


def test_perfect_foresight_beats_every_strategy():
    """On every day, the schedule optimised on the actual prices earns at least as much as a
    schedule optimised on any other price vector (here: noisy and shuffled forecasts)."""
    import pandas as pd

    rng = np.random.default_rng(2)
    rows = []
    for d in range(30):
        n = [23, 24, 25][d % 3]
        actual = rng.normal(70, 50, n)
        for h in range(n):
            rows.append({"ts_utc": d * 100 + h, "delivery_day": d, "price": actual[h]})
    vectors = pd.DataFrame(rows)
    vectors["S-perfect"] = vectors["price"]
    vectors["S-noisy"] = vectors["price"] + rng.normal(0, 40, len(vectors))
    vectors["S-shuffled"] = vectors.groupby("delivery_day")["price"].transform(lambda s: rng.permutation(s.to_numpy()))
    daily = evaluate.daily_profits(vectors, ["S-perfect", "S-noisy", "S-shuffled"])
    assert (daily["S-perfect"] >= daily["S-noisy"] - 1e-6).all()
    assert (daily["S-perfect"] >= daily["S-shuffled"] - 1e-6).all()
    assert (daily["S-perfect"] >= -1e-9).all()  # doing nothing is always feasible


def test_upper_bound_violation_stops_the_run():
    import pandas as pd

    daily = pd.DataFrame({"delivery_day": [1, 2], "S-perfect": [10.0, 20.0], "S-x": [5.0, 21.0]})
    with pytest.raises(AssertionError):
        evaluate.check_perfect_is_upper_bound(daily, ["S-x"])
