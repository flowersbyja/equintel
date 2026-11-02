"""
Melbourne Cup first-four predictor.

Built on top of the equintel ideas (market-as-strong-prior, value = model prob x odds - 1) and the
racing-predictor race-shape / market-adjusted-form ideas, specialised for one race a year:

  * features.py  - tempo-adjusted lead-up form ("did the race shape suit this horse?") and excuse handling
  * patterns.py  - preparation-formula similarity: how closely does this runner's prep resemble the preps of
                   past first-four finishers, with extra credit when it is the SAME trainer's formula and
                   partial credit when a different trainer is using the same style
  * model.py     - log-linear ensemble -> win probabilities -> Plackett-Luce first-four simulation
  * report.py    - markdown report
"""
