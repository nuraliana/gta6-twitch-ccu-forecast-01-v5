# gta6-twitch-ccu-forecast-01-v5

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nuraliana/gta6-twitch-ccu-forecast-01-v5/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nuraliana/gta6-twitch-ccu-forecast-01-v5/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hours 2–24 (UTC), as five quantiles per hour. Python 3.13, standard library only, no network, deterministic.

Built for [make.fun's Forecast GTA VI's launch-day Twitch viewers hour by hour bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model — v5 (target pre-release profile, λ=0.25)

Core = v4, unchanged when both toggles are off: shape learned from earlier console-only launches (no PC, no early access) as log-ratios `ln(viewers_h / viewers_1)`. Centre per hour is the median of three log-ratio-vs-log(p1) fits (OLS, Theil-Sen, grid LAD). The OLS leg's per-hour slopes are smoothed over a 7-hour window and shrunk 50% toward their own mean. Spread is normal quantiles with sigma = max(0.30, IQR of unshrunk OLS residuals / 1.349). Fallback (<3 training games) is the house persistence baseline. Final-only live-signal nudges from v4 are kept as-is (cap ±0.10/hour).

New in v5: the target's own pre-release intraday profile. For each training game, the log-baseline profile per UTC hour is estimated from its hourly pre-release window (14 days before its day_start), day-demeaned, Fourier-smoothed (2 harmonics), and centred. The pool profile is the per-hour median over training games. The target profile comes from the case's `history_hourly` plus, for Grand Theft Auto VI, its own category's 82-day pre-release history (632 points). Centre correction: `centre'[i] = centre[i] + λ · (d_target[i+1] − d_pool[i+1])`, with λ=0.25, reliability-weighted (n/(n+3)). The recursive+direct blend (w) stays 0 — it was tested and rejected decisively.

Validated (production-faithful protocol): LOO-7 0.138741, LOO-37 0.674754, practice 0.186050 — first version to break the 0.140 target on LOO-7, best on all three metrics vs round 2 (LOO-7 0.148027) and v3/v4 (0.146691).

---

<sub>Entered on [make.fun](https://make.fun/?ref=ctnjhg8e&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
