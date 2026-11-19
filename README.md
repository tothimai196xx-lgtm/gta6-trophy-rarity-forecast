[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tothimai196xx-lgtm/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tothimai196xx-lgtm/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — week-one Exophase quantiles

Forecasts, for each of the 55 PlayStation 5 trophies of Grand Theft Auto VI,
its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles
(5 / 25 / 50 / 75 / 95%), from the house's reads of the trophy page up to the
case checkpoint. Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

An earn rate is a ratio, `rate = E / N`: `E` players have earned the trophy
out of `N` players Exophase tracks. The two sides move for different
reasons — `N` jumps whenever Exophase refreshes its counts and pulls in a
backlog of newly tracked players (diluting every rate), while `E` grows as
players actually play — so the program models them separately and divides
at the target:

- `log N` against `log(1 + hours since 2026-11-19T00:00Z)`, least squares
  over the case's distinct reads.
- `log E` per trophy on the same axis, slope shrunk toward zero when the
  history is short, so a short or noisy history falls back toward
  persistence. Median rate at the target = `100 * E(target) / N(target)`,
  kept within 5.5 logit of the last read.
- Flat history (every read identical, as on launch night): no refresh has
  been observed yet, so the last read is stale. The median is shifted down
  1.0 logit — the first refresh on launch night took tracked players
  638 → 3,868 while almost all newly tracked players held only the
  universal first trophy — with a wide spread. The trophy already at 100%
  stays at 100%.
- Spread combines the random-walk term (`0.5 * sqrt(horizon in days)`),
  the fit's residual scatter, and the largest logit jump between
  consecutive distinct reads (a refresh shock the history has shown).

Quantiles are the median on the logit scale plus `z_tau * sigma`.
Python 3.13, standard library only, no network, deterministic.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for each.

## Local measurement

The kit ships no practice answers, so the practice answers were
reconstructed from the figures the rules publish for the 04:30Z refresh
(3,868 players tracked, 4,276 trophies earned): variant A distributes the
new earners (other than the universal 100% trophy) in proportion to each
trophy's launch-night earn count; variant B is pure dilution (no new
earners except the universal trophy). The house states its baseline scores
about 1.51 on the practice.

| forecast | practice loss (A) | practice loss (B) | typical miss (A) | 90% coverage (A) |
| --- | --- | --- | --- | --- |
| house baseline (starter) | 1.4048 | 1.6877 | 0.68 pp | 0.00 |
| this entry | 0.2665 | 0.4363 | 0.13 pp | 1.00 |

Both practice cases score the same (their answers are the same refresh).
Format checks: 55 trophies × 5 non-decreasing quantiles in (0, 100],
two runs byte-identical, runtime ~0.05 s per run. A synthetic final-like
case (power-law growth of `N` and `E` with stepped refreshes, 6,510-minute
horizon) scores 0.346 vs the baseline's 0.449.

Reference: https://make.fun/?ref=wxz47nrz
