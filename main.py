"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Model
-----
An earn rate is a ratio, rate = E / N, where E is how many tracked players
have earned the trophy and N is how many players Exophase tracks.  The two
sides move for different reasons -- N jumps whenever Exophase refreshes its
counts and pulls in a backlog of newly tracked players (which dilutes every
rate), while E grows as players actually play -- so this program models them
separately and divides at the target:

* N(t): least-squares line of log N against log(1 + hours since the release
  day began, 2026-11-19T00:00Z), over the case's distinct reads.
* E_i(t): the same log-log line per trophy, from the trophy's earn counts
  (rate * N / 100), with the slope shrunk toward zero when few distinct
  reads exist, so a short or noisy history falls back toward persistence.
* Median rate at the target = 100 * E_i(target) / N(target), clamped to a
  sane logit move from the last read.

Two regimes the history itself selects:

* Flat history (every read identical, as on launch night): no refresh has
  been observed yet, so the last read is stale.  The next refresh is
  expected to pull in a backlog of newly tracked players and dilute rates
  (on launch night the first refresh took tracked players 638 -> 3,868
  while almost all of the newly tracked players held only the universal
  first trophy, so typical rates fell to roughly a fifth).  The median is
  therefore shifted down 1.0 logit units (more for longer horizons, capped),
  with a wide spread.  A trophy already at 100% is left at 100%: it is the
  universal trophy every tracked player holds, and a refresh does not move it.
* Varying history: the fitted E/N trends above, with a spread that combines
  the house random-walk term (0.5 * sqrt(horizon in days)), the fit's
  residual scatter, and the largest logit jump between consecutive distinct
  reads (a refresh shock the history has actually shown).

Quantiles are the median on the logit scale plus z_tau * sigma, rounded to
4 decimals.  Standard library only, no network, deterministic.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day begins here; the log-time axis is measured from it, as in the
# house curve baseline.
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(at):
    """log(1 + hours since ORIGIN) for a read time."""
    hours = (parse_time(at) - ORIGIN).total_seconds() / 3600.0
    return math.log1p(max(-0.9, hours))


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_reads(history):
    """Collapse consecutive identical reads (same N and same rates)."""
    out = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _key, read in out]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(max(-0.9, (target - ORIGIN).total_seconds() / 3600.0))

    reads = distinct_reads(history)
    last = history[-1]
    flat = len(reads) <= 1

    # --- shared tracked-players model: log N against log time ---
    # Two distinct reads already pin a line; with only two, the slope is
    # still used (clipped) rather than discarded.
    if len(reads) >= 2 and len({r["players_tracked"] for r in reads}) >= 2:
        xs = [log_time(r["at"]) for r in reads]
        a_n, b_n, _resid_n = ols(xs, [math.log(max(1, r["players_tracked"])) for r in reads])
        b_n = min(2.5, max(0.0, b_n))
        log_n_target = a_n + b_n * x_target
    else:
        b_n = 0.0
        log_n_target = math.log(max(1, last["players_tracked"]))

    # Largest logit jump between consecutive distinct reads (refresh shock).
    max_jump = 0.0
    for prev, cur in zip(reads, reads[1:]):
        for i in range(len(trophies)):
            r0, r1 = prev["rates"][i], cur["rates"][i]
            if r0 is None or r1 is None:
                continue
            max_jump = max(max_jump, abs(logit(r1) - logit(r0)))

    rw = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            # Trophy missing from the last read: use its latest known rate.
            known = [r["rates"][i] for r in history if r["rates"][i] is not None]
            last_rate = known[-1] if known else 0.01
        if last_rate >= 99.99:
            # Universal trophy: every tracked player holds it, refreshes
            # included.  Leave it at 100.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        last_logit = logit(last_rate)
        if flat:
            # No refresh observed yet: expect the next one to dilute rates
            # as new tracked players arrive holding few trophies.
            extra = 0.5 * math.log2(max(1.0, horizon_days / 0.0625))
            shift = -min(2.5, 1.0 + extra)
            median_logit = last_logit + shift
            sigma = max(0.6, rw)
        else:
            # Per-trophy earn-count trend on the log-log axis.
            pts = []
            for r in reads:
                rate = r["rates"][i]
                if rate is None:
                    continue
                count = max(0.5, rate * r["players_tracked"] / 100.0)
                pts.append((log_time(r["at"]), math.log(count)))
            if len(pts) >= 2 and len({y for _x, y in pts}) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                a_e, b_e, resid = ols(xs, ys)
                b_e = min(5.0, max(-1.0, b_e))
                # Shrink the slope toward zero when the history is short.
                b_e *= len(pts) / (len(pts) + 3.0)
                a_e = sum(ys) / len(ys) - b_e * (sum(xs) / len(xs))
                log_e_target = a_e + b_e * x_target
                # Two points leave no residual to measure: assume some scatter.
                fit_spread = min(1.5, resid) if len(pts) >= 3 else 0.5
            else:
                log_e_target = math.log(max(0.5, last_rate * last["players_tracked"] / 100.0))
                fit_spread = 0.5
            median_rate = 100.0 * math.exp(log_e_target - log_n_target)
            median_logit = logit(median_rate)
            # A week of launch play can move a rate a long way, but not
            # without bound: keep the median within 5.5 logit of the last read.
            median_logit = min(last_logit + 5.5, max(last_logit - 5.5, median_logit))
            jump_spread = min(1.0, 0.35 * max_jump)
            sigma = math.sqrt(rw * rw + fit_spread * fit_spread + jump_spread * jump_spread + 0.15 * 0.15)

        qs = []
        prev_q = None
        for z in Z:
            q = round4(inv_logit(median_logit + z * sigma))
            q = min(100.0, max(0.0, q))
            if prev_q is not None and q < prev_q:
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
