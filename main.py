"""Forecast GTA VI PS5 trophy earn rates on Exophase, one week after launch.

Model, written fresh for this entry (standard library only, one file):

For every trophy the program reads the case's history -- each Exophase read
(time, earn rate, players tracked) -- and forecasts five quantiles of the
rate at the case's target.

* Earners, not just rates.  A rate is earners / players-tracked.  New players
  arrive with no trophies (dilution pushes rates down) while existing players
  keep earning (rates up).  The program models both sides: it fits a
  saturating curve in log-time to the earners and to the players tracked, and
  reads the ratio at the target, then blends that with a direct saturating
  fit of the rate's logit.
* Saturating shape.  logit(rate) = a + b * ln(1 + t / tau), tau chosen per
  trophy from a grid by weighted least squares (recent reads weigh more, 24 h
  half-life).  Slopes are shrunk toward their tier's mean (platinum / gold /
  silver / bronze) so rare trophies with thin data do not run away.
* Honest spread.  The spread is calibrated by backtesting inside the case's
  own history: fit up to a cutoff, forecast 1.5 / 6 / 24 / 48 hours ahead,
  measure the logit error, and grow the spread with the horizon from those
  errors.  A trophy whose own fit is noisy gets a wider band.
* Frozen pages.  When every read in the history is identical (the page has
  not refreshed since launch), a refresh is coming and it brings a flood of
  new players, so the honest median sits below the last rate and the band is
  wide.

Deterministic, no network, no randomness.  Reads $MAKEFUN_INPUT_DIR (else
/input/), writes $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
TAUS = [1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 168.0, 336.0]
SAT_TAUS = [6.0, 12.0, 24.0, 48.0, 96.0, 168.0, 336.0, 672.0]
CLAMP_LO, CLAMP_HI = 0.01, 99.99


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def clamp_rate(x):
    return min(CLAMP_HI, max(CLAMP_LO, float(x)))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0:
        return 1.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return e / (1.0 + e)


def round4(x):
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, sse, xbar, ybar)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        sse = sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys))
        return (ybar, 0.0, sse, xbar, ybar)
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    return (a, b, sse, xbar, ybar)


def fit_logtime(ts, ys, ws):
    """Fit y = a + b*ln(1 + t/tau) over the tau grid. Returns (tau, a, b, sse).

    The grid is walked from large tau to small so that a tie in fit keeps the
    smoother (larger tau) curve, which extrapolates less wildly.
    """
    best = None
    for tau in reversed(TAUS):
        xs = [math.log1p(t / tau) for t in ts]
        fit = wls(xs, ys, ws)
        if fit is None:
            continue
        a, b, sse, _xbar, _ybar = fit
        if best is None or sse < best[3] * 0.999:
            best = (tau, a, b, sse)
    return best


def fit_saturating(ts, ys, ws):
    """Fit y = a + b*exp(-t/tau): a curve that flattens toward a.

    Unlike the log-time fit this one saturates, so it cannot keep climbing
    forever.  Returns (tau, a, b, sse) or None.
    """
    best = None
    for tau in SAT_TAUS:
        us = [math.exp(-t / tau) for t in ts]
        fit = wls(us, ys, ws)
        if fit is None:
            continue
        a, b, sse, _xbar, _ybar = fit
        if best is None or sse < best[3] * 0.999:
            best = (tau, a, b, sse)
    return best


def recency_weights(ts):
    t_last = max(ts)
    return [0.5 ** ((t_last - t) / 24.0) for t in ts]


def thin(ts, rs, ns, cap=150):
    """Thin a long series to at most cap points, keeping the last one."""
    if len(ts) <= cap:
        return ts, rs, ns
    step = (len(ts) - 1) / (cap - 1)
    keep = sorted(set(round(i * step) for i in range(cap)))
    return ([ts[i] for i in keep], [rs[i] for i in keep],
            [ns[i] for i in keep] if ns is not None else None)


def trophy_series(case, idx):
    """(times_h, rates, players) for one trophy, skipping null rates."""
    ts, rs, ns = [], [], []
    for read in case.get("history", []):
        rates = read.get("rates") or []
        rate = rates[idx] if idx < len(rates) else None
        if rate is None:
            continue
        ts.append(hours_since_release(parse_time(read["at"])))
        rs.append(float(rate))
        ns.append(float(read["players_tracked"]))
    return ts, rs, ns


def players_series(case):
    ts, ns = [], []
    for read in case.get("history", []):
        ts.append(hours_since_release(parse_time(read["at"])))
        ns.append(float(read["players_tracked"]))
    return ts, ns


def last_rate_of(case, idx):
    for read in reversed(case.get("history", [])):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return float(rates[idx])
    return 1.0


def fit_one(ts, rs, ns, t_tgt, tier):
    """Fit one trophy. Returns components for the median and the residual."""
    comp = {"n": len(ts), "tier": tier,
            "last_rate": rs[-1] if rs else 1.0,
            "last_logit": logit(rs[-1]) if rs else logit(1.0),
            "slope": 0.0, "eslope": 0.0, "direct": None, "efit": None,
            "sat": None, "resid": 1.0}
    if len(ts) < 2:
        return comp
    ws = recency_weights(ts)
    sw = sum(ws)
    ys = [logit(r) for r in rs]
    direct = fit_logtime(ts, ys, ws)
    if direct is not None:
        tau, a, b, sse = direct
        xs = [math.log1p(t / tau) for t in ts]
        xbar = sum(w * x for w, x in zip(ws, xs)) / sw
        ybar = sum(w * y for w, y in zip(ws, ys)) / sw
        comp["direct"] = (tau, a, b, sse, xbar, ybar)
        comp["slope"] = b
        comp["resid"] = math.sqrt(max(0.0, sse) / max(1e-9, sw))
    sat = fit_saturating(ts, ys, ws)
    if sat is not None:
        comp["sat"] = sat
    es = [max(0.5, r / 100.0 * n) for r, n in zip(rs, ns)]
    efit = fit_logtime(ts, [math.log(e) for e in es], ws)
    if efit is not None:
        tau_e, a_e, b_e, sse_e = efit
        xs_e = [math.log1p(t / tau_e) for t in ts]
        xbar_e = sum(w * x for w, x in zip(ws, xs_e)) / sw
        ybar_e = sum(w * math.log(e) for w, e in zip(ws, es)) / sw
        comp["efit"] = (tau_e, a_e, b_e, sse_e, xbar_e, ybar_e)
        comp["eslope"] = b_e
    return comp


def median_from(comp, slope, eslope, t_tgt, players_fit, horizon_days):
    """Ensemble median, damped toward the last rate as the horizon grows.

    Members: the flat last rate, the log-time trend, the saturating curve,
    and the earners-over-players structure.  No single form can be trusted
    4.5 days out, so the move from the last rate is shrunk by
    lam = 1 / (1 + horizon_days / 3): a 90-minute forecast trusts its trend,
    a 4.5-day one mostly trusts the last observed level.
    """
    members = [(comp["last_logit"], 0.25)]
    if comp["direct"] is not None:
        tau, _a, _b, _sse, xbar, ybar = comp["direct"]
        a = ybar - slope * xbar
        members.append((a + slope * math.log1p(t_tgt / tau), 0.25))
    if comp["sat"] is not None:
        tau_s, a_s, b_s, _se = comp["sat"]
        members.append((a_s + b_s * math.exp(-t_tgt / tau_s), 0.30))
    if comp["efit"] is not None and players_fit is not None:
        tau_e, _ae, _be, _se, xbar_e, ybar_e = comp["efit"]
        a_e = ybar_e - eslope * xbar_e
        e_tgt = math.exp(a_e + eslope * math.log1p(t_tgt / tau_e))
        tau_n, a_n, b_n, _sn = players_fit
        n_tgt = math.exp(a_n + b_n * math.log1p(t_tgt / tau_n))
        members.append((logit(clamp_rate(100.0 * e_tgt / max(1.0, n_tgt))), 0.20))
    wsum = sum(w for _m, w in members)
    med = sum(m * w for m, w in members) / wsum
    lam = max(0.25, 1.0 / (1.0 + horizon_days / 3.0))
    med = comp["last_logit"] + lam * (med - comp["last_logit"])
    # A 4.5-day horizon cannot honestly move a rate more than this.
    med = min(max(med, comp["last_logit"] - 2.0), comp["last_logit"] + 2.5)
    return min(max(med, logit(CLAMP_LO)), logit(CLAMP_HI))


def fit_all(case, target_time, horizon_hours=108.5, fast=False):
    """Fit every trophy. Returns [{median_logit, resid, ...}] in trophy order."""
    trophies = case.get("trophies", [])
    t_tgt = hours_since_release(target_time)
    horizon_days = max(0.04, horizon_hours / 24.0)
    players_fit = None
    if not fast:
        pts, pns = players_series(case)
        if len(pts) >= 2:
            pts, pns, _ = thin(pts, pns, None)
            pws = recency_weights(pts)
            players_fit = fit_logtime(pts, [math.log(max(1.0, v)) for v in pns], pws)
    comps = []
    for idx, trophy in enumerate(trophies):
        ts, rs, ns = trophy_series(case, idx)
        ts, rs, ns = thin(ts, rs, ns)
        comps.append(fit_one(ts, rs, ns, t_tgt, trophy.get("tier")))
    # Shrink each trophy's slopes toward its tier's mean slope.
    tier_groups = {}
    for c in comps:
        if c["n"] >= 3:
            tier_groups.setdefault(c["tier"], []).append(c)
    tier_slope, tier_eslope = {}, {}
    for tier, cs in tier_groups.items():
        wsum = sum(c["n"] for c in cs)
        if wsum:
            tier_slope[tier] = sum(c["slope"] * c["n"] for c in cs) / wsum
            tier_eslope[tier] = sum(c["eslope"] * c["n"] for c in cs) / wsum
    out = []
    for c in comps:
        slope, eslope = c["slope"], c["eslope"]
        if c["tier"] in tier_slope and c["n"] >= 3:
            slope = (c["n"] * slope + 3.0 * tier_slope[c["tier"]]) / (c["n"] + 3.0)
            eslope = (c["n"] * eslope + 3.0 * tier_eslope[c["tier"]]) / (c["n"] + 3.0)
        med = median_from(c, slope, eslope, t_tgt, players_fit, horizon_days)
        out.append({"median_logit": med, "resid": c["resid"],
                    "last_logit": c["last_logit"]})
    return out


def is_frozen(case):
    """True when every read shows the identical page (no refresh yet)."""
    hist = case.get("history", [])
    if len(hist) < 2:
        return False
    first_rates = hist[0].get("rates")
    first_n = hist[0].get("players_tracked")
    return all(r.get("rates") == first_rates and
               r.get("players_tracked") == first_n for r in hist)


def frozen_forecast(last_rate, horizon_hours):
    """Dilution prior for a page that has not refreshed since launch.

    The next refresh brings a flood of new players (the launch-night page
    sat frozen for hours, then the tracked pool jumped several-fold), so the
    median sits well below the last rate; early trophies that new players
    earn fast fall a little less.  The band is wide: the refresh size is
    very uncertain.
    """
    if last_rate >= 99.5:
        return logit(99.99), 0.15
    players_mult = 5.5
    earners_growth = 1.0 + 0.5 * math.sqrt(max(horizon_hours, 0.25)) * \
        (last_rate / 100.0) ** 0.25
    median_rate = last_rate * earners_growth / players_mult
    sigma = 0.8 + 0.3 * math.sqrt(horizon_hours / 24.0)
    return logit(median_rate), sigma


def backtest_sigma(case, target_time, horizon_hours):
    """Calibrate the spread from the case's own history.

    Fit up to a cutoff, forecast delta ahead, measure the logit error
    against the read nearest the cutoff + delta.  Fit log sigma against
    log delta and extrapolate to the real horizon.
    """
    hist = case.get("history", [])
    if len(hist) < 8:
        return None
    times = [parse_time(r["at"]) for r in hist]
    t_first, t_last = times[0], times[-1]
    span_h = (t_last - t_first).total_seconds() / 3600.0
    if span_h < 24:
        return None
    points = []
    for delta in (1.5, 6.0, 24.0, 48.0):
        errs = []
        for k in range(2):
            cutoff = t_last - timedelta(hours=delta) - timedelta(hours=k * span_h / 6.0)
            if (cutoff - t_first).total_seconds() / 3600.0 < 6:
                continue
            sub = [r for r, t in zip(hist, times) if t <= cutoff]
            if len(sub) < 3:
                continue
            actual_time = cutoff + timedelta(hours=delta)
            actual = next((r for r, t in zip(hist, times) if t >= actual_time), None)
            if actual is None:
                continue
            if (parse_time(actual["at"]) - actual_time).total_seconds() / 3600.0 > 1.5:
                continue
            subcase = dict(case)
            subcase["history"] = sub
            fits = fit_all(subcase, actual_time, delta, fast=True)
            for idx, f in enumerate(fits):
                rates = actual.get("rates") or []
                y = rates[idx] if idx < len(rates) else None
                if y is None:
                    continue
                errs.append(logit(y) - f["median_logit"])
        if len(errs) >= 10:
            rms = math.sqrt(sum(e * e for e in errs) / len(errs))
            points.append((delta, max(0.3, rms)))
    if not points:
        return None
    if len(points) == 1:
        d0, s0 = points[0]
        return max(0.45, min(1.5, s0 * math.sqrt(horizon_hours / d0)))
    xs = [math.log(d) for d, _ in points]
    ys = [math.log(s) for _, s in points]
    fit = wls(xs, ys, [1.0] * len(xs))
    if fit is None:
        return None
    a, beta, _sse, _xb, _yb = fit
    beta = min(0.6, max(0.25, beta))
    sigma = math.exp(a + beta * math.log(horizon_hours))
    return min(1.5, max(0.45, sigma))


def make_row(tid, med_logit, sigma):
    qs = []
    for z in Z:
        q = 100.0 * sigmoid(med_logit + z * sigma)
        qs.append(round4(min(CLAMP_HI, max(CLAMP_LO, q))))
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return {"id": tid, "q": qs}


def forecast_case(case):
    trophies = case.get("trophies", [])
    target_time = parse_time(case["target"])
    horizon_hours = max(1.0, float(case.get("horizon_minutes", 0)) / 60.0)
    if is_frozen(case):
        rows = []
        for idx, trophy in enumerate(trophies):
            med_logit, sigma = frozen_forecast(last_rate_of(case, idx), horizon_hours)
            rows.append(make_row(trophy["id"], med_logit, sigma))
        return rows
    fits = fit_all(case, target_time, horizon_hours)
    sigma_h = backtest_sigma(case, target_time, horizon_hours)
    resids = sorted(f["resid"] for f in fits)
    r_med = resids[len(resids) // 2] if resids else 1.0
    rows = []
    for trophy, f in zip(trophies, fits):
        if sigma_h is None:
            sigma = 0.5 * math.sqrt(horizon_hours / 24.0) + 0.4
        else:
            sigma = sigma_h
        factor = 1.0 + 0.5 * (f["resid"] / max(r_med, 1e-6) - 1.0)
        factor = min(1.8, max(0.7, factor))
        # The backtest extrapolates error growth to a 4.5-day horizon and
        # overstates it; shrink the spread 20% (checked on two synthetic
        # worlds: the loss falls, and the band still covers the truth).
        sigma = min(1.5, max(0.3, sigma * factor * 0.8))
        rows.append(make_row(trophy["id"], f["median_logit"], sigma))
    return rows


def main():
    in_files = []
    if INPUT_DIR.is_dir():
        in_files = sorted(p for p in INPUT_DIR.iterdir()
                          if p.is_file() and p.name.endswith(".in"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in in_files:
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        text = json.dumps({"trophies": rows}, separators=(",", ":"))
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
