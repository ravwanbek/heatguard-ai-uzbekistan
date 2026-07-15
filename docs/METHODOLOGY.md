# Heat-risk methodology

`baseline-v1` is an explainable environmental heat-hazard index, not a medical model. Each normalized input is clipped to a documented range; weighted contributions sum to 0–100.

| Component | Range | Maximum points |
|---|---:|---:|
| Air temperature | 24–43 °C | 24 |
| Apparent temperature | 25–50 °C | 28 |
| Relative humidity | 30–80% | 10 |
| Low wind | inverse, 2–22 km/h | 8 |
| Shortwave radiation | 250–950 W/m² | 8 |
| Land-surface temperature | 30–55 °C | 10 |
| Low NDVI | inverse, 0.1–0.7 | 7 |
| Low elevation | inverse, 0–1,500 m | 3 |
| Built-up fraction | 0.1–0.8 | 2 |

Thresholds are Safe (0–19.9), Moderate (20–39.9), High (40–59.9), Very High (60–79.9), and Extreme (80–100).

The model pipeline learns this rule-based target from historical Open-Meteo weather plus explicitly static city context. It evaluates a future-year holdout; therefore reported metrics describe imitation of this baseline, not medical outcomes. The UI continues to show the transparent rule score in v1, rather than silently replacing it with an ML prediction.
