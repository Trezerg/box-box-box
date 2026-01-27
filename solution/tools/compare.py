#!/usr/bin/env python3
import json
import glob
import sys

try:
    from race_simulator import simulate_race
except ImportError:
    print("Error: Make sure you are running this from the root box-box-box folder!")
    sys.exit(1)

def main():
    files = sorted(glob.glob("data/historical_races/races_*.json"))
    if not files:
        print("No race data found.")
        return

    with open(files[0], "r") as f:
        races = json.load(f)

    print("🔍 DIAGNOSTIC MODE: Checking failures vs Track Temp & Tire Usage\n")
    
    passed = 0
    failed = 0
    
    for race in races[:50]:
        expected = race["finishing_positions"]
        predicted = simulate_race(race["race_config"], race["strategies"])
        
        if predicted == expected:
            passed += 1
            continue
            
        failed += 1
        
        temp = race["race_config"]["track_temp"]
        laps = race["race_config"]["total_laps"]
        
        soft_laps = med_laps = hard_laps = 0
        
        for driver, strategy in race["strategies"].items():
            current_tire = strategy["starting_tire"]
            last_pit_lap = 0
            for pit in strategy.get("pit_stops", []):
                stint = pit["lap"] - last_pit_lap
                if current_tire == "SOFT": soft_laps += stint
                elif current_tire == "MEDIUM": med_laps += stint
                elif current_tire == "HARD": hard_laps += stint
                current_tire = pit["to_tire"]
                last_pit_lap = pit["lap"]
            
            stint = laps - last_pit_lap
            if current_tire == "SOFT": soft_laps += stint
            elif current_tire == "MEDIUM": med_laps += stint
            elif current_tire == "HARD": hard_laps += stint
            
        total = soft_laps + med_laps + hard_laps
        swaps = sum(1 for i in range(20) if expected[i] != predicted[i])
        
        print(f"❌ RACE {race['race_id']} | Temp: {temp}°C | Laps: {laps} | Swaps: {swaps}/20")
        print(f"   Usage -> SOFT: {soft_laps/total:.0%} | MED: {med_laps/total:.0%} | HARD: {hard_laps/total:.0%}\n")
        
    print(f"Stats for first 50 races: {passed} Passed, {failed} Failed")

if __name__ == "__main__":
    main()