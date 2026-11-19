# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) [![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tothimai196xx-lgtm/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) [![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tothimai196xx-lgtm/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — as five quantiles (5/25/50/75/95%)
per trophy — from the house's reads of the trophy page up to the deadline.
Built for the [make.fun bounty](https://make.fun/?ref=wxz47nrz).

## Method

An earn rate is a ratio, so the two sides are modelled separately:

- **E** = players who have earned the trophy, **N** = players Exophase
  tracks. Both are fitted against time on a log-log scale (log E and log N
  each against log(1 + hours since 2026-11-19T00:00:00Z)) by least squares
  over the case's history. The median forecast is 100 · E(target)/N(target).
  Splitting the ratio matters because Exophase refreshes in jumps: a refresh
  brings a wave of newly tracked players who have earned little yet, so N
  jumps faster than E and rates fall while E still grows.
- **Flat history** (no read has moved a trophy's rate yet, as on launch
  night): the next refresh is still ahead, so the median is shifted down
  1.0 logit from the last rate, with a 0.6-logit spread. A trophy at 100%
  stays at 100%.
- **Spread**: a random walk growing with the horizon (0.5 logit per
  square-root day), the residual of the log-log fits, and the largest logit
  jump between two consecutive reads — the refresh size the page has already
  shown it can make — combined in quadrature.

`main.py` is Python 3.13, standard library only, offline and deterministic:
two runs on the same case write byte-identical output.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory and writes one `<id>.out` per
case to the output directory (defaults `/input` and `/output` when the
environment variables are unset).

## Local measurements

Practice cases p00/p01 straddle Exophase's first refresh (638 → 3,868
players tracked, 943 → 4,276 trophies earned). The refreshed answer was
reconstructed from those published figures two ways:

- **A** — the 100% trophy keeps all 3,868 earners; the remaining 408 earned
  trophies are shared by the other 54 trophies in proportion to their
  launch-night earn counts (totals add up to exactly 4,276).
- **B** — pure dilution: earner counts unchanged, only N grows to 3,868.

Mean pinball loss on the logit scale (lower is better), p00:

| model | variant A | variant B |
|---|---|---|
| this entry | **0.2656** | **0.4354** |
| house baseline (kit starter) | 1.4048 | 1.6877 |

90% coverage is 1.00 under both variants. A synthetic final-like test
(power-law growth with stepwise refreshes, 6,510-minute horizon, 120 reads)
gives 0.2624 for this entry vs 0.2727 for the house baseline.

Format checks pass on both practice cases (55 trophies × 5 quantiles,
non-decreasing, in (0, 100]); two runs are byte-identical; runtime is
about 0.07 s for both practice cases.

## License

MIT — see [LICENSE](LICENSE).
