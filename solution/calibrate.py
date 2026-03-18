#!/usr/bin/env python3
import argparse
import glob
import json
import math
import random
import datetime
from pathlib import Path
from typing import List, Tuple

from race_simulator import ModelParams, TireParams, _default_model_params, _simulate_race_with_params

def _rank_score(predicted: List[str], expected: List[str]) -> float:
    # Gradient-guided exact match:
    # 1.0 for a perfect match, plus a tiny 0.01 pairwise signal so the
    # optimizer can feel the slope and isn't flying completely blind.
    exact_match = 1.0 if predicted == expected else 0.0

    n = len(expected)
    pred_pos = {driver: i for i, driver in enumerate(predicted)}
    pairwise_correct = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if pred_pos[expected[i]] < pred_pos[expected[j]]
    )
    pairwise = pairwise_correct / (n * (n - 1) / 2)

    return exact_match + (0.01 * pairwise)

def _evaluate(races: List[dict], params: ModelParams) -> float:
    return sum(
        _rank_score(
            _simulate_race_with_params(r["race_config"], r["strategies"], params),
            r["finishing_positions"]
        ) for r in races
    ) / max(len(races), 1)

def _clamp_params(params: ModelParams) -> ModelParams:
    clamped = params.clone()

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for tire in ("SOFT", "MEDIUM", "HARD"):
        p = clamped.tire_model[tire]
        clamped.tire_model[tire] = TireParams(
            base_delta    = clamp(p.base_delta,     -5.0,  5.0),
            deg_linear    = clamp(p.deg_linear,      0.0,  0.50),
            deg_quadratic = clamp(p.deg_quadratic,   0.0,  0.05),
            temp_deg_coeff= clamp(p.temp_deg_coeff, -0.05, 0.05),
            threshold     = p.threshold  # Always locked
        )
    clamped.temp_reference_c = clamp(params.temp_reference_c, 10.0, 60.0)
    return clamped

def _mutate(params: ModelParams, rng: random.Random, scale: float) -> ModelParams:
    p    = params.clone()
    tire = rng.choice(["SOFT", "MEDIUM", "HARD"])

    # Four parameters per tire + one global temp_reference
    what = rng.choice(["base", "lin", "quad", "temp_deg", "temp_ref"])

    if what == "temp_ref":
        p.temp_reference_c += rng.gauss(0.0, 2.0 * scale)
    else:
        tp = p.tire_model[tire]
        if what == "base":
            tp = TireParams(tp.base_delta + rng.gauss(0.0, 0.25 * scale),
                            tp.deg_linear, tp.deg_quadratic, tp.temp_deg_coeff, tp.threshold)
        elif what == "lin":
            tp = TireParams(tp.base_delta,
                            tp.deg_linear + rng.gauss(0.0, 0.010 * scale),
                            tp.deg_quadratic, tp.temp_deg_coeff, tp.threshold)
        elif what == "quad":
            tp = TireParams(tp.base_delta, tp.deg_linear,
                            tp.deg_quadratic + rng.gauss(0.0, 0.0005 * scale),
                            tp.temp_deg_coeff, tp.threshold)
        elif what == "temp_deg":
            tp = TireParams(tp.base_delta, tp.deg_linear, tp.deg_quadratic,
                            tp.temp_deg_coeff + rng.gauss(0.0, 0.0015 * scale),
                            tp.threshold)
        p.tire_model[tire] = tp

    return _clamp_params(p)

def _reservoir_sample_historical(rng: random.Random, sample_size: int, data_glob: str) -> List[dict]:
    selected, seen = [], 0
    files = sorted(glob.glob(data_glob))
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            races = json.load(f)
        for race in races:
            seen += 1
            if len(selected) < sample_size:
                selected.append(race)
            elif rng.randrange(seen) < sample_size:
                selected[rng.randrange(sample_size)] = race
    return selected

def _optimize(races: List[dict], initial: ModelParams, iterations: int, seed: int) -> Tuple[ModelParams, float]:
    rng           = random.Random(seed)
    current       = _clamp_params(initial)
    current_score = _evaluate(races, current)
    best_score    = current_score
    best          = current.clone()

    for i in range(1, iterations + 1):
        progress        = i / max(iterations, 1)
        candidate       = _mutate(current, rng, 1.5 - 1.2 * progress)
        candidate_score = _evaluate(races, candidate)

        if (candidate_score >= current_score or
                rng.random() < math.exp(
                    (candidate_score - current_score) /
                    max(0.001, 0.04 * (1.0 - progress))
                )):
            current, current_score = candidate, candidate_score

        if current_score > best_score:
            best, best_score = current.clone(), current_score

        if i % 100 == 0 or i == iterations:
            print(f"iter={i:5d}  current={current_score:.6f}  best={best_score:.6f}")

    return best, best_score

def save_calibration_results(
    best: ModelParams, score: float, samples: int,
    path_json: Path, path_log: Path
) -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dict = {
        "TIMESTAMP":       now_str,
        "BEST_SCORE":      score,
        "SAMPLES_USED":    samples,
        "TIRE_MODEL": {
            k: {
                "base_delta":     v.base_delta,
                "deg_linear":     v.deg_linear,
                "deg_quadratic":  v.deg_quadratic,
                "temp_deg_coeff": v.temp_deg_coeff,
                "threshold":      v.threshold,
            }
            for k, v in best.tire_model.items()
        },
        "TEMP_REFERENCE_C": best.temp_reference_c,
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4)
    with open(path_log, "a", encoding="utf-8") as f:
        f.write(f"Calibration Run: {now_str}\nBest score: {score:.6f}\n")

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--iterations",  type=int, default=2000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data-glob",   default="data/historical_races/races_*.json")

    import sys
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    root   = Path(__file__).resolve().parent.parent
    races  = _reservoir_sample_historical(
        random.Random(args.seed), args.sample_size, str(root / args.data_glob)
    )
    best, best_score = _optimize(races, _default_model_params(), args.iterations, args.seed)
    save_calibration_results(
        best, best_score, len(races),
        root / "solution/model_parameters.json",
        root / "solution/calibration_log.txt"
    )

if __name__ == "__main__":
    main()
