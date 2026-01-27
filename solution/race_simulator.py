#!/usr/bin/env python3
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class TireParams:
    base_delta: float
    deg_linear: float
    deg_quadratic: float
    age_temp_interaction: float
    threshold: float

@dataclass
class ModelParams:
    tire_model: Dict[str, TireParams]
    temp_reference_c: float
    temp_sensitivity: Dict[str, float]

    def clone(self) -> "ModelParams":
        return ModelParams(
            tire_model={k: TireParams(v.base_delta, v.deg_linear, v.deg_quadratic, v.age_temp_interaction, v.threshold) for k, v in self.tire_model.items()},
            temp_reference_c=self.temp_reference_c,
            temp_sensitivity=dict(self.temp_sensitivity)
        )

# Sensible priors with LOCKED CLIFFS
TIRE_MODEL: Dict[str, TireParams] = {
    "SOFT": TireParams(-1.00, 0.08, 0.002, 0.001, 10.0),
    "MEDIUM": TireParams(-0.40, 0.04, 0.001, 0.0005, 20.0),
    "HARD": TireParams(0.20, 0.02, 0.0005, 0.0001, 30.0),
}
TEMP_REFERENCE_C = 35.0
TEMP_SENSITIVITY = {"SOFT": 0.015, "MEDIUM": 0.010, "HARD": 0.005}

def load_parameters() -> None:
    global TIRE_MODEL, TEMP_REFERENCE_C, TEMP_SENSITIVITY
    config_path = Path(__file__).parent / "model_parameters.json"
    hardcoded_cliffs = {"SOFT": 10.0, "MEDIUM": 20.0, "HARD": 30.0}
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("TIRE_MODEL", {}).items():
                TIRE_MODEL[k] = TireParams(
                    v["base_delta"], 
                    v["deg_linear"], 
                    v["deg_quadratic"], 
                    v["age_temp_interaction"], # Restored
                    hardcoded_cliffs.get(k, 0.0)
                )
            TEMP_REFERENCE_C = data.get("TEMP_REFERENCE_C", TEMP_REFERENCE_C)
            sens = data.get("TEMP_SENSITIVITY", {})
            for k, v in sens.items():
                TEMP_SENSITIVITY[k] = v
        except Exception:
            pass

load_parameters()

def _default_model_params() -> ModelParams:
    return ModelParams(
        tire_model={k: TireParams(v.base_delta, v.deg_linear, v.deg_quadratic, v.age_temp_interaction, v.threshold) for k, v in TIRE_MODEL.items()},
        temp_reference_c=TEMP_REFERENCE_C,
        temp_sensitivity=dict(TEMP_SENSITIVITY)
    )

def _build_pit_map(pit_stops: List[dict]) -> Dict[int, str]:
    return {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}

def _simulate_driver_with_params(race_config: dict, strategy: dict, params: ModelParams) -> float:
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = strategy["starting_tire"]
    pit_map = _build_pit_map(strategy.get("pit_stops", []))

    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        
        p = params.tire_model[current_tire]
        
        # 1. Temperature affects the BASE speed of the fresh tire
        temp_delta = track_temp - params.temp_reference_c
        temp_effect = params.temp_sensitivity[current_tire] * temp_delta
        
        # 2. Base Degradation Calculation (Post-Cliff)
        effective_age = max(0.0, float(tire_age) - p.threshold)
        degradation = p.deg_linear * effective_age + p.deg_quadratic * (effective_age * effective_age)
        
        # 3. Temperature accelerates degradation (Interaction)
        interaction = p.age_temp_interaction * effective_age * temp_delta
        
        lap_time = base_lap_time + p.base_delta + temp_effect + degradation + interaction
        total_time += lap_time

        if lap in pit_map:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time

def _simulate_race_with_params(race_config: dict, strategies: dict, params: ModelParams) -> List[str]:
    totals: List[Tuple[float, str]] = []
    for pos_key, strategy in strategies.items():
        driver_id = strategy["driver_id"]
        race_time = _simulate_driver_with_params(race_config, strategy, params)
        totals.append((race_time, driver_id))

    # Deterministic Tie-Breaker
    totals.sort(key=lambda item: (item[0], item[1]))
    return [driver_id for _, driver_id in totals]

def simulate_race(race_config: dict, strategies: dict) -> List[str]:
    return _simulate_race_with_params(race_config, strategies, _default_model_params())

def _validate_output(finishing_positions: List[str]) -> None:
    if len(finishing_positions) != 20:
        raise ValueError("Must contain exactly 20 drivers")

def main():
    try:
        input_data = sys.stdin.read()
        if not input_data.strip(): return
        test_case = json.loads(input_data)
        finishing_positions = simulate_race(test_case["race_config"], test_case["strategies"])
        _validate_output(finishing_positions)
        print(json.dumps({"race_id": test_case["race_id"], "finishing_positions": finishing_positions}))
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()