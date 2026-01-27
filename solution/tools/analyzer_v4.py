import json
from scipy.optimize import differential_evolution

# --- THE HONEST F1 PHYSICS ENGINE ---
def simulate_race(race, params):
    config = race['race_config']
    base_time = config['base_lap_time']
    pit_penalty = config['pit_lane_time']
    temp = config['track_temp']
    total_laps = config['total_laps']
    
    # Extract our 8 unknown variables
    offset_S, offset_H, deg_S, deg_M, deg_H, temp_base, temp_coeff, fuel_rate = params
    
    offsets = {"SOFT": offset_S, "MEDIUM": 0.0, "HARD": offset_H}
    deg_rates = {"SOFT": deg_S, "MEDIUM": deg_M, "HARD": deg_H}
    cliffs = {"SOFT": 10, "MEDIUM": 20, "HARD": 30} # Discovered from ML script
    
    results = []
    
    for pos_key, driver in race['strategies'].items():
        total_time = 0.0
        current_tire = driver['starting_tire']
        tire_age = 0
        pit_stops = {stop['lap']: stop['to_tire'] for stop in driver['pit_stops']}
        
        for lap in range(1, total_laps + 1):
            tire_age += 1 
            
            # 1. Tire Degradation (Hockey Stick)
            deg_multiplier = max(0, tire_age - cliffs[current_tire])
            temp_factor = temp_base + (temp_coeff * temp)
            wear_loss = deg_multiplier * deg_rates[current_tire] * temp_factor
            
            # 2. Fuel Weight (Car gets lighter/faster as laps decrease)
            fuel_weight_penalty = (total_laps - lap) * fuel_rate
            
            lap_time = base_time + offsets[current_tire] + wear_loss + fuel_weight_penalty
            total_time += lap_time
            
            # 3. Pit stop at the END of the lap
            if lap in pit_stops:
                total_time += pit_penalty
                current_tire = pit_stops[lap]
                tire_age = 0 
            
        results.append((total_time, driver['driver_id']))
        
    # TIEBREAKER RULE: Sort by Time (Ascending), then Driver ID (Ascending)
    results.sort(key=lambda x: (x[0], x[1]))
    return [r[1] for r in results]

# --- THE ERROR SCORER ---
def calculate_error(params, races):
    total_error = 0
    for race in races:
        predicted = simulate_race(race, params)
        actual = race['finishing_positions']
        
        for driver_id in actual:
            diff = abs(predicted.index(driver_id) - actual.index(driver_id))
            total_error += diff
            
    return total_error

def main():
    print("Loading historical data...")
    try:
        # Load 50 races for a robust sample size
        with open('../data/historical_races/races_08000-08999.json', 'r') as f:
            races = json.load(f)[:50] 
    except FileNotFoundError:
        print("Error: Could not find the JSON file.")
        return

    # Boundaries for our 8 unknown variables
    bounds = [
        (-2.0, -0.1),  # SOFT offset (faster, so negative)
        (0.1, 2.0),    # HARD offset (slower, so positive)
        (0.01, 0.5),   # SOFT deg_rate
        (0.005, 0.2),  # MEDIUM deg_rate
        (0.001, 0.1),  # HARD deg_rate
        (-2.0, 2.0),   # temp_base
        (-0.2, 0.2),   # temp_coeff
        (0.01, 0.2)    # fuel_burn_rate (time lost per lap of fuel weight)
    ]

    print("Starting V5 Honest Physics Optimizer... (Hunting for 8 variables)")
    
    def objective(params):
        return calculate_error(params, races)
        
    # Run the Genetic Algorithm
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=50, 
        popsize=15, 
        mutation=(0.5, 1.0), 
        recombination=0.7,
        seed=42,
        disp=True # Prints progress every generation!
    )

    print("\n==================================================")
    print(f"🏁 OPTIMIZATION COMPLETE - LOWEST ERROR: {result.fun}")
    print("==================================================")
    print(f"OFFSET_SOFT:  {result.x[0]:.4f}")
    print(f"OFFSET_HARD:  {result.x[1]:.4f}")
    print(f"DEG_SOFT:     {result.x[2]:.6f}")
    print(f"DEG_MEDIUM:   {result.x[3]:.6f}")
    print(f"DEG_HARD:     {result.x[4]:.6f}")
    print(f"TEMP_BASE:    {result.x[5]:.6f}")
    print(f"TEMP_COEFF:   {result.x[6]:.6f}")
    print(f"FUEL_RATE:    {result.x[7]:.6f}")

if __name__ == "__main__":
    main()