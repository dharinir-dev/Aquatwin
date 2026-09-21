# AquaTwin AI — Machine Learning Module

This folder adds a real, trained Random Forest classifier to the AquaTwin AI
prototype, so the "AI" in AquaTwin AI is an actual model rather than fixed
if/else thresholds.

## What's here

```
aquatwin_ml/
├── dataset/
│   └── sensor_training_data.csv   <- dummy training data (18 rows, 3 classes)
├── models/
│   └── aquatwin_rf_model.joblib   <- created after you run train_model.py
├── train_model.py                 <- trains and saves the classifier
├── predict.py                     <- runs demo predictions + judge-facing text
├── serial_bridge.py               <- OPTIONAL: live Arduino <-> model link
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python train_model.py      # trains the Random Forest, saves it to models/
python predict.py          # runs 3 built-in demo readings through the model
```

To test a specific reading manually:
```bash
python predict.py 26.1 57 33 -21.0 0.3
# args: temp_c  humidity_pct  water_level_pct  delta_level  delta_humidity
```

## How the model works

Five features per reading:
- `temp_c` — current temperature
- `humidity_pct` — current relative humidity
- `water_level_pct` — current water level
- `delta_level` — change in level since the last reading (negative = drop)
- `delta_humidity` — change in humidity since the last reading

Three classes: **Normal**, **Evaporation**, **Leak** — matching the dummy
dataset's pattern:
- **Normal**: level and humidity roughly flat
- **Evaporation**: level drops *gradually* while humidity *rises*
- **Leak**: level drops *suddenly* while humidity stays flat (no rise)

The Random Forest learns these patterns from the 18 example rows in
`dataset/sensor_training_data.csv` (6 per class) and generalizes to new
readings it hasn't seen — that's the genuine ML story to tell judges,
versus a hardcoded rule.

## What to say to judges

Here's a version of your line that matches what the model actually does
(your original phrasing had the leak condition backwards — humidity should
stay flat on a leak, not rise, since a leak loses water without putting
moisture into the air the way evaporation does):

> "The model has learned the normal sensor pattern from training data. When
> the water level drops suddenly *without* the expected rise in humidity, it
> recognizes this combination as an abnormal leak signature and recommends
> diverting the water flow and triggering the alert immediately. A gradual
> drop *with* rising humidity is instead classified as evaporation, which
> triggers the recovery cycle rather than an alert."

If you'd rather keep your original wording as the two-condition story for
simplicity, that's a presentation choice — just be ready for a technical
judge to ask why humidity would rise on a leak, since physically it
shouldn't.

## Connecting it to the live prototype (optional)

`serial_bridge.py` is a ready-to-use bridge: it reads live sensor lines
from your Arduino Uno over USB serial, runs them through the trained model,
and writes a single character back (`N`/`E`/`L`) so your Arduino sketch can
react — e.g. trip the relay on `L`. This requires:
1. Your Arduino sketch printing one CSV line per reading over Serial, in
   the exact order: `temp_c,humidity_pct,water_level_pct,delta_level,delta_humidity`
2. Your Arduino sketch listening for an incoming byte and acting on
   `'N'`/`'E'`/`'L'`
3. `pip install pyserial`, then update `SERIAL_PORT` in the script to match
   your system

This is optional — for the competition demo, running `predict.py` on a
laptop next to the physical prototype and narrating the output is enough to
show real ML in action without needing to debug live serial integration
under time pressure.

## Honest caveats (say these proactively if asked)

- The dataset is small, hand-crafted dummy data for demonstration — not
  logged from extended real-world operation. That's fine for a
  proof-of-concept but should be named as a known limitation and next step.
- With more time, the natural upgrade is logging real sensor sequences
  from the prototype across many run cycles (including edge cases and
  sensor noise) and retraining on that.
