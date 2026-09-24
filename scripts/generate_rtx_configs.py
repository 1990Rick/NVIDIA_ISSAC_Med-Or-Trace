#!/usr/bin/env python3
"""Generate Isaac Sim RTX lidar/radar JSON profiles from configs/sensors/sensor_rig.yaml.

The lidar JSON follows the Isaac Sim RTX lidar profile layout
(class/type/profile with emitterStates).  Radar follows the generic RTX radar
profile layout.  Isaac Sim versions differ in accepted keys; run
``scripts/isaac/validate_sensor_configs.py`` inside Isaac Sim to check the
profiles against the installed version before large runs.
"""
import json

import _bootstrap  # noqa: F401
import numpy as np

from medortrace.common.config import CONFIG_DIR, load_yaml

cfg = load_yaml(CONFIG_DIR / "sensors" / "sensor_rig.yaml")
L = cfg["lidar"]
n = int(L["channels"])
el = np.linspace(L["elevation_deg"][0], L["elevation_deg"][1], n)
fire_ns = (np.arange(n) * 3000 // n).astype(int)          # staggered firing within a 3 us burst
steps_per_rev = int(round(360.0 / L["azimuth_resolution_deg"]))
lidar = {
    "class": "sensor",
    "type": "lidar",
    "name": L["name"],
    "driveWorksId": "GENERIC",
    "profile": {
        "scanType": "rotary",
        "intensityProcessing": "normalization",
        "rayType": "IDEALIZED",
        "nearRangeM": L["near_range_m"],
        "farRangeM": L["far_range_m"],
        "startAzimuthDeg": 0.0,
        "endAzimuthDeg": 360.0,
        "upElevationDeg": float(max(el)),
        "downElevationDeg": float(min(el)),
        "rangeResolutionM": L["range_resolution_m"],
        "rangeAccuracyM": L["range_accuracy_m"],
        "avgPowerW": L["avg_power_w"],
        "minReflectance": L["min_reflectance"],
        "minReflectanceRange": L["min_reflectance_range_m"],
        "wavelengthNm": L["wavelength_nm"],
        "pulseTimeNs": L["pulse_time_ns"],
        "azimuthErrorMean": 0.0, "azimuthErrorStd": 0.01,
        "elevationErrorMean": 0.0, "elevationErrorStd": 0.01,
        "maxReturns": L["max_returns"],
        "scanRateBaseHz": L["scan_rate_hz"],
        "reportRateBaseHz": int(L["scan_rate_hz"] * steps_per_rev),
        "numberOfEmitters": n,
        "emitterStates": [{
            "azimuthDeg": [0.0] * n,
            "elevationDeg": [round(float(e), 4) for e in el],
            "fireTimeNs": [int(f) for f in fire_ns],
        }],
        "intensityMappingType": "LINEAR",
    },
}
R = cfg["radar"]
radar = {
    "class": "sensor",
    "type": "radar",
    "name": R["name"],
    "profile": {
        "carrierFrequencyGHz": R["carrier_ghz"],
        "bandwidthGHz": R["bandwidth_ghz"],
        "maxRangeM": R["max_range_m"],
        "rangeResolutionM": R["range_resolution_m"],
        "maxVelocityMps": R["max_velocity_mps"],
        "velocityResolutionMps": R["velocity_resolution_mps"],
        "azimuthFovDeg": R["fov_azimuth_deg"],
        "elevationFovDeg": R["fov_elevation_deg"],
        "azimuthResolutionDeg": R["azimuth_resolution_deg"],
        "frameRateHz": R["frame_rate_hz"],
        "scanType": "SRR",
        "outputs": ["range", "azimuth", "elevation", "radialVelocity", "rcs"],
    },
}
out = CONFIG_DIR / "sensors"
(out / "rtx_lidar_or16.json").write_text(json.dumps(lidar, indent=2) + "\n")
(out / "rtx_radar_or77.json").write_text(json.dumps(radar, indent=2) + "\n")
print("wrote", out / "rtx_lidar_or16.json", out / "rtx_radar_or77.json")
