"""Synthetic IoT / time-series data generator.

Produces readings that carry both a numeric value and a natural-language note,
mirroring the "simulate an IoT sensor dataset" Tiger Data tutorial. The notes
are written to be semantically searchable so vector similarity is meaningful.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

SITES = ["plant-a", "plant-b", "warehouse-1", "edge-gateway"]
SENSORS = {
    "cpu_temperature": ("C", 30.0, 95.0),
    "vibration": ("mm/s", 0.1, 12.0),
    "pressure": ("bar", 0.8, 6.5),
    "humidity": ("%", 20.0, 90.0),
    "power_draw": ("W", 50.0, 900.0),
    "network_latency": ("ms", 2.0, 400.0),
}

# (template, tags) — tags nudge the note toward a searchable theme.
ANOMALY_NOTES = [
    ("{sensor} on {device} spiked sharply; readings consistent with a cooling failure or blocked airflow at {site}", "cooling-failure"),
    ("{sensor} on {device} climbed steadily through the shift, matching a sustained heavy batch job at {site}", "batch-job"),
    ("{sensor} on {device} shows a periodic oscillation typical of bearing wear and mechanical imbalance at {site}", "bearing-wear"),
    ("{sensor} on {device} drifted above baseline after a firmware update; possible calibration regression at {site}", "firmware-regression"),
    ("{sensor} on {device} recorded intermittent dropouts correlating with network congestion at {site}", "network-congestion"),
    ("{sensor} on {device} spiked right after power restoration, likely an inrush transient at {site}", "power-transient"),
    ("{sensor} on {device} fell below the safe threshold during an unplanned maintenance window at {site}", "unsafe-threshold"),
    ("{sensor} on {device} jumped during peak production hours, pointing to thermal throttling at {site}", "thermal-throttling"),
]

NORMAL_NOTES = [
    ("{sensor} on {device} stayed within normal operating range at {site}", "nominal"),
    ("{sensor} on {device} held steady with only minor natural variation at {site}", "nominal"),
    ("{sensor} on {device} read a healthy value typical for this time of day at {site}", "nominal"),
    ("{sensor} on {device} settled back to baseline after a small transient at {site}", "recovered"),
]

DEVICES = {
    "plant-a": ["press-01", "press-02", "robot-arm-03", "conveyor-07"],
    "plant-b": ["cnc-11", "cnc-12", "chiller-02", "boiler-01"],
    "warehouse-1": ["hvac-01", "dock-door-04", "forklift-09", "freezer-03"],
    "edge-gateway": ["gw-01", "gw-02", "gw-03", "router-05"],
}


def generate(n: int = 4000, days: int = 7, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(days=days)
    span = (now - start).total_seconds()

    rows: list[dict] = []
    for _ in range(n):
        ts = start + timedelta(seconds=rng.uniform(0, span))
        site = rng.choice(SITES)
        device = rng.choice(DEVICES[site])
        sensor_type, (unit, lo, hi) = rng.choice(list(SENSORS.items()))

        anomaly = rng.random() < 0.28
        if anomaly:
            value = rng.uniform(hi * 0.85, hi * 1.15)
            template, _tag = rng.choice(ANOMALY_NOTES)
        else:
            value = rng.uniform(lo, hi * 0.7)
            template, _tag = rng.choice(NORMAL_NOTES)

        note = template.format(sensor=sensor_type.replace("_", " "), device=device, site=site)
        rows.append(
            {
                "ts": ts,
                "device_id": device,
                "site": site,
                "sensor_type": sensor_type,
                "value": round(value, 3),
                "unit": unit,
                "note": note,
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return rows
