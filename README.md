# Equintel

Horse racing prediction system using XGBoost. Predicts win/place probability for each runner and identifies value bets by comparing model fair odds to bookmaker odds.

**Saudi adaptation in progress** — see `docs/SAUDI_ADAPTATION.md` and `src/saudi_loader.py` for JCSA / Riyadh / Taif data sources and loader skeleton.

## How it works

1. **Historical data** — Kaggle SQLite dataset (~466 MB, UK/Ireland 2015-2026). Saudi path uses structured JCSA results (CSV/JSON) or commercial APIs.
2. **Feature engineering** — Form ratings, jockey/trainer stats, going preference, market implied probability, draw position, days since last run, and more. Saudi extensions: surface, local-bred, temperature.
3. **Model** — Two calibrated XGBoost classifiers: one for win, one for place.
4. **Live inference** — The Racing API (UK) or future Saudi card source.
5. **Value detection** — Edge = (model probability × bookie odds) − 1; flag where edge > threshold (research mode).

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the historical data

**UK/Ireland:**
Download from Kaggle (free account required):
[Horse Racing results - UK/Ireland 2015-2026](https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025)

Extract `raceform.db` into the `data/` directory:

```
data/raceform.db
```

**Saudi:**
See `docs/SAUDI_ADAPTATION.md`. Use `src/saudi_loader.py` with a normalized CSV of JCSA results.

### 3. Set up The Racing API (UK path)

Sign up for a free account at [theracingapi.com](https://www.theracingapi.com), then:

```bash
cp .env.example .env
# Edit .env and add your credentials
```

### 4. Build features

This step processes the raw database and computes all form/stats features. Takes ~1-2 hours on the full dataset (11 years of races).

```bash
python scripts/build_features.py
```

For a quick test with a subset:

```bash
python scripts/build_features.py --limit 100000
```

### 5. Train the models

```bash
python scripts/train_model.py
```

This trains win and place models with an 18-month validation holdout and prints AUC / Brier scores.

### 6. Predict today's races

```bash
python scripts/predict_today.py
```

Options:

```bash
python scripts/predict_today.py --date 2026-04-12
python scripts/predict_today.py --course Cheltenham
python scripts/predict_today.py --min-edge 10 --max-odds 20
python scripts/predict_today.py --show-all  # Show full predictions per race
```

## Features used by the model

| Category | Features |
|----------|----------|
| Race context | Distance (furlongs), going (ordinal), field size, race class |
| Runner | Age, weight, draw position, official rating, RPR, top speed |
| Market | Log SP, implied probability (bookmaker signal is strong) |
| Horse form | Win rate overall / at distance / on going / at course, last 3 positions, days since last run |
| Jockey | 90-day win rate and place rate |
| Trainer | 90-day win rate and place rate |

## Output interpretation

```
horse          win_prob  place_prob  fair_win_odds  fair_place_odds  bookie_odds
Dashing Star   18.3%     52.1%       5.5            1.9              7.0
Running Brook  14.1%     43.2%       7.1            2.3              6.0
```

**Value bets table:**

```
horse         bet_type  bookie_odds  fair_odds  model_prob  edge_pct  kelly_stake_pct
Dashing Star  win       7.0          5.5        18.3        27.8      6.2
```

- `edge_pct` > 0 = positive expected value
- `kelly_stake_pct` = recommended stake as % of bankroll (half-Kelly, conservative)

## Project structure

```
equintel/
├── data/                   # gitignored — put raceform.db here
├── models/                 # trained model artifacts (gitignored)
├── docs/
│   └── SAUDI_ADAPTATION.md   # Saudi data sources and adaptation notes
├── src/
│   ├── data_loader.py      # Load/parse SQLite DB
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training pipeline
│   ├── predict.py          # Inference for a single racecard
│   ├── api_client.py       # The Racing API wrapper
│   ├── value.py            # Value bet identification
│   └── saudi_loader.py     # Saudi JCSA results loader skeleton
├── scripts/
│   ├── build_features.py   # CLI: compute feature dataset
│   ├── train_model.py      # CLI: train models
│   └── predict_today.py    # CLI: predict and find value bets
├── requirements.txt
├── .env.example
└── README.md
```

## Melbourne Cup first-four predictor

A one-race-a-year module (`melbourne_cup/`) that reuses the equintel ideas (market as a strong prior, value = model probability x odds - 1) and the racing-predictor race-shape ideas, and adds two things the generic models do not have:

1. **Tempo-adjusted lead-up form** (`melbourne_cup/features.py`) - a finishing position is read against how the lead-up was run: a backmarker beaten in a slowly run race is upgraded, a closer that won off a slow tempo is discounted, on-pacers that faded in a genuinely run race are excused, and recognised excuses (first-up, wide, heavy track, interference, dropped whip, big weight) add merit back. Short sharpeners are down-weighted.
2. **Preparation-formula similarity** (`melbourne_cup/patterns.py`) - every runner is compared with the preps of all first-four finishers since 2000 (`melbourne_cup/cup_data/historical_cups.json`): lead-up race, days between runs, number of Australian runs, origin, age, weight, barrier, running style. A match to the **same trainer's** past Cup prep scores full credit (Joseph O'Brien's Irish St Leger -> Cup route, Maher's Caulfield Cup -> Cox Plate -> Cup route); the same style used by a **different trainer** scores 65% credit, so a horse can be flagged as "Rekindling-like" even when it is not trained by Rekindling's trainer.

3. **Trainer methodology profiles** (`melbourne_cup/cup_data/trainer_styles.json`) - each trainer is described on eight dimensions (fresh vs race-hardened, Australian runs, European base, weight-for-age sharpeners, forward tactics, lightly raced young horses, Golden Ticket targeting, data-driven). The profile is blended with how the individual horse was actually campaigned, so Maher's Point Of Law (trained from Werribee quarantine with one run) reads as fly-in shaped even though Maher's norm is 3-5 runs. A match to another trainer's past Cup horse is credited at 0.35 + 0.65 x method similarity.
4. **Going model** (`melbourne_cup/going.py`) - a distribution over the expected rating, and per-horse wet records blended with connections' comments.
5. **Race shape** (`melbourne_cup/pace.py`) - speed map from predicted settling positions, pace-pressure read (racing-predictor thresholds), a running-style history prior, and a Cup Day bias switch (`--bias on-pace|closers`) for when the early races show which part of the field is winning.

The pieces are combined log-linearly with a shrunk market prior and small structural priors (weight, barrier, age, sex, trainer and jockey Cup record, slowly run last lead-up), then a Plackett-Luce model turns win probabilities into place / top-4 probabilities and the most likely exact and boxed first fours. `--form-guide` writes a multi-variable form guide with a card per runner.

```bash
python scripts/predict_melbourne_cup.py                                   # print the report
python scripts/predict_melbourne_cup.py --out docs/MELBOURNE_CUP_2026.md --json docs/melbourne_cup_2026_predictions.json
python scripts/predict_melbourne_cup.py --track soft6 --bias on-pace --form-guide docs/MELBOURNE_CUP_2026_FORM_GUIDE.md
python scripts/predict_melbourne_cup.py --market-weight 0.6 --pattern-weight 0.4 --sims 100000
python -m pytest tests/test_melbourne_cup.py
```

The 2026 field, odds snapshot, lead-ups and tempo notes live in `melbourne_cup/cup_data/field_2026.json`; edit the odds and `expected_track` on race morning and re-run. The 2026 report is in `docs/MELBOURNE_CUP_2026.md`.

## Notes

- The market (SP) is a strong predictor — the model finds edge by combining it with form features the market doesn't fully price in
- Use half-Kelly or smaller stakes; the Kelly Criterion assumes perfect probability estimates
- This is for research and educational purposes
- Saudi path prioritizes official JCSA sources and evidence-style records over pure betting output
