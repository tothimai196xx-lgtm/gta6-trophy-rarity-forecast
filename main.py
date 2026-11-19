"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's trophies, five quantiles of its earn rate
on Exophase at the case's target, from the reads in the case's history.

Method, in brief
----------------
A trophy's earn rate is a ratio, so the two sides of it are modelled
separately: E, the number of tracked players who have earned the trophy, and
N, the number of players Exophase tracks.  Both are fitted against time on a
log-log scale -- log E and log N each against log(1 + hours since
2026-11-19T00:00:00Z) -- by least squares over the case's history, and the
median forecast at the target is 100 * E(target) / N(target).  Splitting the
ratio matters because Exophase refreshes its counts in jumps: a refresh
brings in a wave of newly tracked players who have earned little yet, so N
jumps faster than E and the rate falls even while E keeps growing.

Flat history.  When a trophy's rate has not moved at all across the history
(launch night: every read shows the same page), no refresh has been seen yet,
so the next one is still ahead of us.  A refresh dilutes rates, so the median
is shifted down by 1.0 logit from the last rate, with a spread of 0.6 logit.
A trophy sitting at 100% stays at 100%.

Spread.  The logit spread around the median combines three sources, added in
quadrature: a random walk that grows with the horizon (0.5 logit per
square-root day, the house baseline's scale), the residual of the log-log
fits, and the largest logit jump between two consecutive reads in the
history -- the size of refresh the page has already shown it can make.

Only the Python standard library is used.  The program reads every ``*.in``
file in the input directory (``$MAKEFUN_INPUT_DIR`` or ``/input``) and writes
one ``<id>.out`` per case to the output directory (``$MAKEFUN_OUTPUT_DIR`` or
``/output``).  It is deterministic: two runs on the same case write the same
bytes.  It never touches the network.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Time origin of the log-log fits: the start of release day, UTC.
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Flat-history branch: how far a not-yet-seen refresh is expected to push a
# rate down the logit scale, and how uncertain that push is.
FLAT_SHIFT_LOGIT = -1.0
FLAT_SIGMA = 0.6

# Random-walk spread, in logit units per square-root day of horizon.
RW_SIGMA = 0.5

# Smallest spread allowed, so a perfectly smooth history never claims a
# forecast is exact.
MIN_SIGMA = 0.2


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x: float) -> float:
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float:
    """Round to 4 decimals, half up (the way the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return float(value)


def least_squares(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (intercept, slope, rms residual).

    With fewer than two distinct x values there is no line: returns a flat
    line through the mean of y with zero residual.
    """
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        resid = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n) if n else 0.0
        return mean_y, 0.0, resid
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(
        sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n
    )
    return intercept, slope, resid


def forecast_trophy(series, horizon_days, target_x):
    """Forecast one trophy from its (x, E, N, rate) history rows.

    ``series`` holds one row per read that listed the trophy, oldest first.
    Returns the five quantiles, in percent, non-decreasing.
    """
    last_rate = series[-1][3]

    # A trophy every tracked player holds stays held: 100% throughout.
    if last_rate >= 100.0 and all(row[3] >= 100.0 for row in series):
        return [100.0, 100.0, 100.0, 100.0, 100.0]

    rw = RW_SIGMA * math.sqrt(horizon_days)

    # Largest logit jump between two consecutive reads: the refresh size the
    # page has already demonstrated.
    jump = 0.0
    for prev, cur in zip(series, series[1:]):
        step = abs(logit_pct(cur[3]) - logit_pct(prev[3]))
        if step > jump:
            jump = step

    flat = all(row[3] == last_rate for row in series)
    xs = [row[0] for row in series]
    distinct_x = len(set(xs)) > 1

    if flat or not distinct_x or len(series) < 2:
        # No refresh seen yet: expect the next one to dilute the rate.
        median_logit = logit_pct(last_rate) + FLAT_SHIFT_LOGIT
        sigma = math.sqrt(FLAT_SIGMA ** 2 + rw ** 2)
    else:
        # Log-log fits of E and N against time; both counts only grow, so a
        # negative fitted slope is sampling noise and is clipped to flat.
        log_e = [math.log(max(row[1], 0.5)) for row in series]
        log_n = [math.log(max(row[2], 1.0)) for row in series]
        a_e, b_e, r_e = least_squares(xs, log_e)
        a_n, b_n, r_n = least_squares(xs, log_n)
        b_e = max(0.0, b_e)
        b_n = max(0.0, b_n)
        e_tgt = math.exp(a_e + b_e * target_x)
        n_tgt = math.exp(a_n + b_n * target_x)
        median_rate = 100.0 * e_tgt / n_tgt if n_tgt > 0 else last_rate
        median_logit = logit_pct(median_rate)
        # Fit residual (both sides of the ratio) plus the demonstrated jump.
        resid = math.sqrt(r_e ** 2 + r_n ** 2)
        sigma = math.sqrt(rw ** 2 + resid ** 2 + jump ** 2)

    sigma = max(sigma, MIN_SIGMA)

    quantiles = []
    prev_q = 0.01
    for z in Z:
        q = round4(inv_logit_pct(median_logit + z * sigma))
        q = min(100.0, max(0.01, q))
        # Rounding must never break the non-decreasing order.
        if q < prev_q:
            q = prev_q
        quantiles.append(q)
        prev_q = q
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = (parse_time(case["target"]) - ORIGIN).total_seconds() / 3600.0
    # Guarded the same way as the history x values: a target before the
    # origin still yields a usable log-time.
    target_x = math.log(max(1.0 + target_hours, 0.01))

    # One (x, E, N, rate) row per read, per trophy, oldest first.
    per_trophy = [[] for _ in case["trophies"]]
    for read in case["history"]:
        t = parse_time(read["at"])
        hours = (t - ORIGIN).total_seconds() / 3600.0
        x = math.log(max(1.0 + hours, 0.01))
        n_players = read["players_tracked"]
        for i, rate in enumerate(read["rates"]):
            if rate is None or n_players is None:
                continue
            earners = rate / 100.0 * n_players
            per_trophy[i].append((x, earners, float(n_players), float(rate)))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = per_trophy[i]
        if not series:
            # No read ever listed this trophy: fall back to a flat 1% guess
            # with a wide spread rather than failing the whole case.
            series = [(0.0, 0.5, 1.0, 1.0)]
        rows.append({"id": trophy["id"], "q": forecast_trophy(series, horizon_days, target_x)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
