"""Forecast GTA VI launch-day Twitch CCU, hours 2-24, five quantiles/hour.

Method (v5):
- Core = v4, unchanged and byte-identical when both new toggles are
  off (lambda=0, w_rec=0): shape learned from earlier console-only
  launches (no PC, no early access) as log-ratios ln(viewers_h /
  viewers_1). Centre per hour is the median of three
  log-ratio-vs-log(p1) fits (OLS, Theil-Sen, grid LAD). The OLS leg's
  per-hour slopes are smoothed over a 7-hour window and shrunk 50%
  toward their own mean. Spread is normal quantiles with sigma =
  max(0.30, IQR of unshrunk OLS residuals / 1.349). Fallback (<3
  training games) is the house persistence baseline. Final-only
  live-signal nudges from v4 are kept as-is (cap +/-0.10/hour).
- NEW, Idea 1 (RESEARCH-V5.md): pre-release intraday profile of the
  target's own category (baseline + surge). For each training game,
  the log-baseline profile per UTC hour B_g[h] is estimated from its
  hourly pre-release window (14 days before its day_start), day-
  demeaned so pre-launch hype trends do not contaminate the hour
  profile, then Fourier-smoothed (2 harmonics) and centred to
  d_g[h] = B_g[h] - mean_h(B_g[h]). The pool profile d_pool[h] is
  the per-hour median over training games. The target profile
  d_target[h] comes from the case's `history_hourly` (practice;
  in LOO the lab injects the held-out game's own pre-release
  hourly there) plus, when the target IS Grand Theft Auto VI, the
  "Grand Theft Auto VI category before release" comparable (82
  days, 632 points -- its day_start is the Aug-27 premiere, so
  its hourly rows are used as-is, not filtered by its own
  day_start). Centre correction: centre'[i] = centre[i] +
  lambda * (d_target[i+1] - d_pool[i+1]), i = 0..22, where
  i+1 is the UTC hour of forecast hour i+2 (day_start is
  00:00Z). lambda in {0, 0.25, 0.5}; lambda=0 disables it.
  FASE 4 (Kandidat A, reliability-weighting): the per-hour
  correction is shrunk by n/(n+3), where n is the number of
  qualifying days behind the target's bin at that UTC hour
  (k = PROFILE_MIN_DAYS, the code's own minimum-days
  threshold). Thin bins -- one event-day can dominate the
  bin mean -- shrink toward the pool (no correction); thick
  bins are trusted. This replaces the unweighted correction
  above; lambda and w_rec are unchanged, w_rec stays 0.
- NEW, Idea 2 (RESEARCH-V5.md): recursive (adjacent-hour growth)
  + direct blend. From each training curve, step growths
  g[k] = ln(v_{k+1}) - ln(v_k), k = 0..22, are run through the
  SAME median-of-3 machine as the direct centre (same slope
  smoothing window 7 + shrink 0.5, same per-step guard +/-1.5).
  The recursive centre R is the cumulative sum of the predicted
  steps, with the cumulative deviation from the mean-step
  cumulative sum also guarded at +/-1.5, then smoothed window 3.
  Final centre = (1-w)*direct + w*R, w in {0, 0.25, 0.5};
  w=0 disables it. Sigma stays v3 (direct residuals).
- NOT built, Idea 3 (RESEARCH-V5.md): log-p1 extrapolation
  saturation. It cannot be validated by LOO/practice; it is a
  judgment call for Nur/parent, not a lab decision.
Toggles: forecast()/run_case() take lam= / w_rec= arguments;
the CLI reads GTA6_V5_LAMBDA / GTA6_V5_WREC from the environment
(defaults lam=0.25, w_rec=0.0, clamped to [0, 1]). With both at 0 the output is
byte-identical to v4 on every case.
Stdlib only, no network, deterministic.
"""
import glob
import json
import math
import os
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
FLOOR = 0.30
SPREAD_MULT = 1.0
GUARD = 1.5
SHRINK = 0.5
SLOPE_WIN = 7
SMOOTH_WIN = 3
NUDGE_CAP = 0.10
GTA_HISTORY_PREFIX = "Grand Theft Auto VI category before release"
PROFILE_DAYS = 14
PROFILE_MIN_BINS = 12
PROFILE_MIN_DAYS = 3


def _parse_time(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _mean(values):
    return sum(values) / len(values)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _quantile(ordered, q):
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _smooth(values, window):
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        out.append(_mean(values[lo:hi + 1]))
    return out


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _day_curve(comp):
    """24 hourly viewer counts for the release day; None where missing."""
    try:
        start = _parse_time(comp.get("day_start", ""))
    except Exception:
        return [None] * 24
    slots = {}
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row.get("from", ""))
            viewers = row.get("viewers")
        except Exception:
            continue
        k = int((when - start).total_seconds() // 3600)
        if 0 <= k < 24 and isinstance(viewers, (int, float)):
            slots[k] = viewers
    return [slots.get(k) for k in range(24)]


def _training_game_comps(case):
    """Console-only launches without early access and with a full curve."""
    games = []
    for comp in case.get("comparables") or []:
        name = comp.get("name", "") or ""
        if name.startswith(GTA_HISTORY_PREFIX):
            continue
        if "PC" in (comp.get("platforms", "") or ""):
            continue
        if "early access" in (comp.get("unlock", "") or "").lower():
            continue
        curve = _day_curve(comp)
        if curve[0] is None or curve[0] <= 0:
            continue
        if any(v is None or v <= 0 for v in curve):
            continue
        games.append(comp)
    return games


def _training_games(case):
    games = []
    for comp in _training_game_comps(case):
        curve = _day_curve(comp)
        games.append((curve[0], curve))
    return games


def _log_ratios(train):
    xs = [math.log(max(p1, 1.0)) for p1, _ in train]
    ratios = []
    for _, curve in train:
        base = math.log(max(curve[0], 1.0))
        ratios.append([math.log(max(curve[h], 1.0)) - base
                       for h in range(1, 24)])
    return xs, ratios


def _step_ratios(train):
    """Adjacent-hour log growths g[k] = ln(v_{k+1}) - ln(v_k), k = 0..22."""
    ratios = []
    for _, curve in train:
        logs = [math.log(max(v, 1.0)) for v in curve]
        ratios.append([logs[k + 1] - logs[k] for k in range(23)])
    return ratios


def _ols_slopes(xs, ratios):
    n = len(xs)
    xbar = _mean(xs)
    ybar = [_mean([ratios[i][h] for i in range(n)]) for h in range(23)]
    denom = sum((v - xbar) ** 2 for v in xs)
    slopes = []
    for h in range(23):
        if denom <= 1e-12:
            slopes.append(0.0)
            continue
        num = sum((xs[i] - xbar) * (ratios[i][h] - ybar[h])
                  for i in range(n))
        slopes.append(num / denom)
    return ybar, slopes


def _theil_sen(xs, ys):
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(xs[j] - xs[i]) > 1e-12:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        return 0.0, _mean(ys)
    slope = _median(slopes)
    intercept = _median([ys[i] - slope * xs[i] for i in range(n)])
    return intercept, slope


def _lad(xs, ys):
    n = len(xs)
    best = None
    for step in range(-100, 101):
        slope = step / 100.0
        intercept = _median([ys[i] - slope * xs[i] for i in range(n)])
        cost = sum(abs(ys[i] - (intercept + slope * xs[i]))
                   for i in range(n))
        if best is None or cost < best[0]:
            best = (cost, intercept, slope)
    return best[1], best[2]


def _centre(xs, ratios, x):
    """Median of the OLS / Theil-Sen / LAD log-ratio predictions."""
    n = len(xs)
    xbar = _mean(xs)
    ybar, raw_slopes = _ols_slopes(xs, ratios)
    smoothed = _smooth(raw_slopes, SLOPE_WIN)
    global_slope = _mean(raw_slopes)
    shrunk = [(1.0 - SHRINK) * s + SHRINK * global_slope
              for s in smoothed]
    ols = [ybar[h] + _clamp(shrunk[h] * (x - xbar), GUARD)
           for h in range(23)]

    theil = []
    lad = []
    for h in range(23):
        ys = [ratios[i][h] for i in range(n)]
        a, b = _theil_sen(xs, ys)
        theil.append(ybar[h] + _clamp(a + b * x - ybar[h], GUARD))
        a, b = _lad(xs, ys)
        lad.append(ybar[h] + _clamp(a + b * x - ybar[h], GUARD))
    theil = _smooth(theil, SMOOTH_WIN)
    lad = _smooth(lad, SMOOTH_WIN)

    return [sorted([ols[h], theil[h], lad[h]])[1] for h in range(23)]


def _recursive_centre(xs, steps, x):
    """Centre from cumulative adjacent-hour growth predictions.

    Same median-of-3 machine as _centre, applied per step (guard
    +/-1.5 per step inside _centre). The cumulative deviation from
    the mean-step cumulative sum is also guarded at +/-1.5, then
    the cumulative curve is smoothed window 3 like the direct
    centre.
    """
    n = len(xs)
    mean_steps = [_mean([steps[i][k] for i in range(n)])
                  for k in range(23)]
    pred_steps = _centre(xs, steps, x)
    raw = []
    acc = 0.0
    accbar = 0.0
    for k in range(23):
        acc += pred_steps[k]
        accbar += mean_steps[k]
        raw.append(accbar + _clamp(acc - accbar, GUARD))
    return _smooth(raw, SMOOTH_WIN)


def _sigmas(xs, ratios, x):
    """Per-hour spread from unshrunk OLS residuals (window-3 slopes)."""
    n = len(xs)
    xbar = _mean(xs)
    ybar, raw_slopes = _ols_slopes(xs, ratios)
    slopes = _smooth(raw_slopes, SMOOTH_WIN)
    sigmas = []
    for h in range(23):
        column = sorted(
            ratios[i][h] - (ybar[h] + _clamp(slopes[h] * (xs[i] - xbar),
                                             GUARD))
            for i in range(n))
        iqr = _quantile(column, 0.75) - _quantile(column, 0.25)
        sigmas.append(max(FLOOR, SPREAD_MULT * iqr / 1.349))
    return sigmas


def _solve_linear(matrix, rhs):
    """Small dense linear solve (Gaussian elimination, partial pivot)."""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j]
                                  for j in range(i + 1, n))) / aug[i][i]
    return out


def _fill_circular(values):
    """Fill None entries by circular linear interpolation."""
    present = [i for i, v in enumerate(values) if v is not None]
    if not present:
        return [0.0] * len(values)
    out = list(values)
    for i, v in enumerate(values):
        if v is not None:
            continue
        prev_i = max([j for j in present if j < i],
                     default=present[-1] - len(values))
        next_i = min([j for j in present if j > i],
                     default=present[0] + len(values))
        prev_v = values[prev_i % len(values)]
        next_v = values[next_i % len(values)]
        span = next_i - prev_i
        out[i] = prev_v + (next_v - prev_v) * (i - prev_i) / span \
            if span else prev_v
    return out


def _fourier_smooth(values):
    """Circular smoothing with 1-2 harmonics (least squares, K=2).

    Returns the fitted 24 values, or None if the fit fails (caller
    falls back to a circular moving average).
    """
    n = len(values)
    basis = []
    for h in range(n):
        a1 = 2.0 * math.pi * h / n
        a2 = 4.0 * math.pi * h / n
        basis.append([1.0, math.cos(a1), math.sin(a1),
                      math.cos(a2), math.sin(a2)])
    k = 5
    normal = [[sum(basis[h][i] * basis[h][j] for h in range(n))
               for j in range(k)] for i in range(k)]
    for i in range(k):
        normal[i][i] += 1e-9
    rhs = [sum(basis[h][i] * values[h] for h in range(n))
           for i in range(k)]
    beta = _solve_linear(normal, rhs)
    if beta is None:
        return None
    return [sum(basis[h][j] * beta[j] for j in range(k))
            for h in range(n)]


def _profile_from_rows(rows, cutoff, max_days=None):
    """Mean-zero intraday log-baseline deviation d[0..23], or None.

    Returns (profile, counts): profile is the list of 24 centred
    deviations; counts[h] is the number of qualifying days behind
    bin h (one entry per day that contributed a demeaned value to
    that bin). Counts are the reliability input for the Fase 4
    shrinkage in _profile_deviation: a bin resting on few days is
    mostly one event-day, not a diurnal pattern.

    rows: (datetime, viewers) pairs. Only rows before `cutoff`
    (and within `max_days` before it, when given) are used. Each
    UTC day is demeaned first (two-way detrending: day level out,
    hour profile kept), so pre-launch hype trends do not
    contaminate the hour profile. Days with <2 points carry no
    hour contrast and are skipped. The 24-bin profile is
    circularly filled where a bin is empty and Fourier-smoothed
    (2 harmonics; moving-average fallback).
    """
    by_day = {}
    for when, viewers in rows:
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        if when >= cutoff:
            continue
        if max_days is not None and \
                (cutoff - when).total_seconds() > max_days * 86400:
            continue
        logv = math.log(max(float(viewers), 1.0))
        by_day.setdefault(when.date(), {}).setdefault(
            when.hour, []).append(logv)
    per_hour = [[] for _ in range(24)]
    ndays = 0
    for hours in by_day.values():
        means = {h: _mean(v) for h, v in hours.items()}
        if len(means) < 2:
            continue
        ndays += 1
        day_mean = _mean(list(means.values()))
        for h, m in means.items():
            per_hour[h].append(m - day_mean)
    n_bins = sum(1 for col in per_hour if col)
    if ndays < PROFILE_MIN_DAYS or n_bins < PROFILE_MIN_BINS:
        return None
    counts = [len(col) for col in per_hour]
    base = [(_mean(per_hour[h]) if per_hour[h] else None)
            for h in range(24)]
    filled = _fill_circular(base)
    smooth = _fourier_smooth(filled)
    if smooth is None:
        # Fallback: circular moving average, window 5.
        smooth = [_mean([filled[(h + d) % 24] for d in (-2, -1, 0, 1, 2)])
                  for h in range(24)]
    centre = _mean(smooth)
    return [v - centre for v in smooth], counts


def _reliability_weight(n_days):
    """Shrinkage weight for one profile bin from its day count.

    n/(n+k) with k = PROFILE_MIN_DAYS (3): a bin backed by exactly
    the minimum number of days gets 50% weight, 14 days keep 82%,
    the final GTA VI bins (15-30 days) keep 83-91%. Standard
    inverse-variance / empirical-Bayes form: the bin mean's noise
    scales as 1/n, so the correction is shrunk toward the pool
    (zero) where n is small. A bin with no direct data (n=0,
    filled by interpolation) gets weight 0: no evidence, no
    correction at that hour. Note the smoothed profile value at
    hour h also borrows from neighbouring bins, so raw per-bin
    counts are a conservative proxy for its reliability -- the
    conservative direction (shrink more, not less).
    """
    return n_days / (n_days + PROFILE_MIN_DAYS)


def _hourly_rows(comp):
    """(datetime, viewers) pairs from a comparable's hourly series."""
    rows = []
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row.get("from", ""))
            viewers = row.get("viewers")
        except Exception:
            continue
        if isinstance(viewers, (int, float)):
            rows.append((when, viewers))
    return rows


def _profile_deviation(case, train_comps):
    """Per-hour centre correction, reliability-weighted, list of 23,
    or None.

    Index i (0..22) is forecast hour i+2, i.e. UTC hour i+1, so the
    profile bin used is i+1. Returns None when either side cannot
    be estimated (caller then applies no correction). The raw
    deviation d_target - d_pool is shrunk per hour by
    _reliability_weight(n_target[i+1]) (Fase 4, Kandidat A):
    bins whose target mean rests on few days shrink toward the
    pool (no correction); thick bins are trusted. The pool side is
    a median over games, so its per-hour reliability is roughly
    uniform across hours and is not weighted.
    """
    try:
        day_start = _parse_time(case.get("day_start", ""))
    except Exception:
        return None
    pool_cols = [[] for _ in range(24)]
    n_games = 0
    for comp in train_comps or []:
        try:
            own_start = _parse_time(comp.get("day_start", ""))
        except Exception:
            continue
        prof, _ = _profile_from_rows(_hourly_rows(comp), own_start,
                                     max_days=PROFILE_DAYS)
        if prof is None:
            continue
        n_games += 1
        for h in range(24):
            pool_cols[h].append(prof[h])
    if n_games < 3:
        return None
    d_pool = [_median(pool_cols[h]) for h in range(24)]

    rows = []
    seen = set()
    for row in case.get("history_hourly") or []:
        try:
            when = _parse_time(row.get("from", ""))
            viewers = row.get("viewers")
        except Exception:
            continue
        key = row.get("from")
        if isinstance(viewers, (int, float)) and viewers > 0 \
                and when < day_start and key not in seen:
            seen.add(key)
            rows.append((when, viewers))
    target_name = (case.get("game") or {}).get("name", "") or ""
    if "Grand Theft Auto VI" in target_name:
        # The target-history comparable IS the target's own
        # pre-release history (its day_start is the Aug-27
        # premiere, so its rows are taken as-is, all before the
        # Nov-19 launch day_start).
        for comp in case.get("comparables") or []:
            if (comp.get("name", "") or "").startswith(
                    GTA_HISTORY_PREFIX):
                for when, viewers in _hourly_rows(comp):
                    key = when.isoformat()
                    if viewers > 0 and when < day_start \
                            and key not in seen:
                        seen.add(key)
                        rows.append((when, viewers))
    if not rows:
        return None
    d_target, n_target = _profile_from_rows(rows, day_start,
                                            max_days=None)
    if d_target is None:
        return None
    return [_reliability_weight(n_target[i + 1])
            * (d_target[i + 1] - d_pool[i + 1]) for i in range(23)]


def _live_nudges(case, p1):
    """Final-only live-signal nudges to the log-centre, per hour 2-24.

    Returns a list of 23 values, each in [-0.10, +0.10]. All inputs
    are null in practice cases, in which case this returns zeros.
    Four mechanism-based components (see EXPERIMENTS.md for the
    reasoning and the unvalidated-risk caveat):
      1. streamer concentration: top-20 streams' share of hour-1
         viewers. >0.7 -> late hours up to +0.05 (a concentrated
         audience rides big marathon streamers, sustaining the tail);
         <0.5 -> down to -0.05. Ramps in over hours 10-14.
      2. stream freshness: share of top-20 viewers in streams that
         started <=2h before the checkpoint. >0.3 -> early hours up
         to +0.04 (streamers still joining = rising momentum),
         fading out by hour 10.
      3. category-share momentum: target category's share of the
         top-20 categories now vs ~6h ago. A drift of +/-0.20 maps
         to +/-0.05 on all hours.
      4. minute momentum: mean viewers of the last 60 minutes vs
         the 60 minutes before, from minute `history` (needs >=30
         points in the last 2h). Log slope clamped to +/-0.30 maps
         to +/-0.04 on hours 2-6, fading out by hour 10.
    Every component is defensive: missing/malformed data disables
    that component only, never the forecast.
    """
    nudges = [0.0] * 23
    try:
        checkpoint = _parse_time(case.get("checkpoint", ""))
    except Exception:
        return nudges
    p = max(float(p1), 1.0)

    # 1 + 2: streams at the checkpoint.
    try:
        streams = case.get("streams") or []
        total = 0.0
        fresh = 0.0
        for s in streams:
            v = s.get("viewers")
            if not isinstance(v, (int, float)) or v <= 0:
                continue
            total += v
            try:
                started = _parse_time(s.get("started_at", ""))
            except Exception:
                continue
            if 0 <= (checkpoint - started).total_seconds() <= 2 * 3600:
                fresh += v
        if total > 0:
            conc = total / p
            if conc > 0.7:
                amp = 0.05 * min(1.0, (conc - 0.7) / 0.3)
            elif conc < 0.5:
                amp = -0.05 * min(1.0, (0.5 - conc) / 0.3)
            else:
                amp = 0.0
            for h in range(23):
                hour = h + 2
                ramp = 0.0 if hour < 10 else (hour - 10) / 4.0 if hour < 14 else 1.0
                nudges[h] += amp * ramp
            fresh_share = fresh / total
            if fresh_share > 0.3:
                amp2 = 0.04 * min(1.0, (fresh_share - 0.3) / 0.3)
                for h in range(23):
                    hour = h + 2
                    fade = 1.0 if hour <= 6 else max(0.0, (10 - hour) / 4.0)
                    nudges[h] += amp2 * fade
    except Exception:
        pass

    # 3: category-share momentum.
    try:
        cats = case.get("categories_hourly") or []
        target = (case.get("game") or {}).get("name", "") or ""
        shares = []
        for entry in cats:
            try:
                when = _parse_time(entry.get("from", ""))
            except Exception:
                continue
            top = entry.get("top") or []
            tot = sum(r.get("viewers", 0) for r in top
                      if isinstance(r.get("viewers"), (int, float)))
            own = 0.0
            for r in top:
                if r.get("name") == target and isinstance(
                        r.get("viewers"), (int, float)):
                    own = r["viewers"]
            if tot > 0:
                shares.append((when, own / tot))
        if len(shares) >= 2:
            shares.sort()
            now_share = shares[-1][1]
            six_h_ago = checkpoint.timestamp() - 6 * 3600
            past = [s for w, s in shares if w.timestamp() <= six_h_ago]
            if past:
                drift = now_share - past[-1]
                amp3 = 0.05 * _clamp(drift / 0.20, 1.0)
                for h in range(23):
                    nudges[h] += amp3
    except Exception:
        pass

    # 4: minute momentum from history.
    try:
        hist = case.get("history") or []
        pts = []
        for r in hist:
            try:
                when = _parse_time(r.get("at", ""))
                v = r.get("viewers")
            except Exception:
                continue
            if isinstance(v, (int, float)) and v > 0:
                pts.append((when, v))
        pts.sort()
        recent = [v for w, v in pts
                  if 0 <= (checkpoint - w).total_seconds() <= 2 * 3600]
        if len(recent) >= 30:
            last = [v for w, v in pts
                    if 0 <= (checkpoint - w).total_seconds() <= 3600]
            prev = [v for w, v in pts
                    if 3600 < (checkpoint - w).total_seconds() <= 7200]
            if last and prev:
                slope = math.log(_mean(last) / _mean(prev))
                amp4 = 0.04 * _clamp(slope / 0.30, 1.0)
                for h in range(23):
                    hour = h + 2
                    fade = 1.0 if hour <= 6 else max(0.0, (10 - hour) / 4.0)
                    nudges[h] += amp4 * fade
    except Exception:
        pass

    return [_clamp(v, NUDGE_CAP) for v in nudges]


def _rows_from_centre_sigma(p, centre, sigmas):
    rows = []
    for h in range(23):
        q = [p * math.exp(centre[h] + z * sigmas[h]) for z in Z]
        q = [min(max(v, 1.0), 1e9) for v in q]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append(q)
    return rows


def _starter_rows(p):
    rows = []
    for h in range(2, 25):
        spread = 0.15 * math.sqrt(h - 1)
        q = [p * math.exp(z * spread) for z in Z]
        q = [min(max(v, 1.0), 1e9) for v in q]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append(q)
    return rows


def _env_float(name, default=0.0):
    try:
        value = float(os.environ.get(name, "") or default)
    except Exception:
        return default
    if value != value:
        return default
    return max(0.0, min(1.0, value))


def forecast(p1, train, case=None, lam=0.0, w_rec=0.0, train_comps=None):
    p = max(float(p1), 1.0)
    if len(train) < 3:
        return _starter_rows(p)
    xs, ratios = _log_ratios(train)
    x = math.log(p)
    centre = _centre(xs, ratios, x)
    sigmas = _sigmas(xs, ratios, x)
    if w_rec:
        steps = _step_ratios(train)
        rec = _recursive_centre(xs, steps, x)
        centre = [(1.0 - w_rec) * centre[h] + w_rec * rec[h]
                  for h in range(23)]
    if lam and case is not None:
        corr = _profile_deviation(case, train_comps)
        if corr is not None:
            centre = [centre[h] + lam * corr[h] for h in range(23)]
    if case is not None:
        nudges = _live_nudges(case, p)
        centre = [centre[h] + nudges[h] for h in range(23)]
    return _rows_from_centre_sigma(p, centre, sigmas)


def run_case(case, lam=None, w_rec=None):
    if lam is None:
        lam = _env_float("GTA6_V5_LAMBDA", 0.25)
    if w_rec is None:
        w_rec = _env_float("GTA6_V5_WREC", 0.0)
    known = case.get("known") or []
    p1 = 1.0
    if known:
        viewers = known[0].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            p1 = viewers
    comps = _training_game_comps(case)
    train = []
    for comp in comps:
        curve = _day_curve(comp)
        train.append((curve[0], curve))
    rows = forecast(p1, train, case=case, lam=lam, w_rec=w_rec,
                    train_comps=comps if lam else None)
    return {"hours": [{"hour": h + 2, "q": rows[h]} for h in range(23)]}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path) as handle:
            case = json.load(handle)
        out = run_case(case)
        base = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out_dir, base + ".out"), "w") as handle:
            json.dump(out, handle)


if __name__ == "__main__":
    main()
