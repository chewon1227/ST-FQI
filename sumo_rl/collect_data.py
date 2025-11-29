import argparse
import os
import sys
# Add project root to sys.path to allow running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
from sumo_rl import SumoEnvironment
from sumo_rl.gangnam_utils import GangnamObservationFunction, gangnam_reward

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Data Collection for Gangnam Intersection (FQI)"
    )
    prs.add_argument("-net", dest="net_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int2.net.xml", help="Network file")
    prs.add_argument("-route", dest="route_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int_scaled.rou.xml", help="Route file")
    prs.add_argument("-seconds", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-delta", dest="delta_time", type=int, default=60, help="Delta time (cycle length)")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-out", dest="out_csv_name", type=str, default="gangnam_data.csv", help="Output CSV filename.\n")
    prs.add_argument("--ts_id", dest="ts_id", type=str, default=None, help="Specific Traffic Signal ID to control (e.g. J0). If None, controls all.\n")
    prs.add_argument("--teleport", dest="teleport", type=int, default=-1, help="Time to teleport stuck vehicles (seconds). -1 to disable.\n")
    
    args = prs.parse_args()
    
    ts_ids = [args.ts_id] if args.ts_id else None

    print(f"Initializing environment with net: {args.net_file}")
    env = SumoEnvironment(
        net_file=args.net_file,
        route_file=args.route_file,
        out_csv_name=None, # We save manually
        use_gui=False, # Original value, as args.gui is not defined by the instruction
        num_seconds=args.seconds,
        delta_time=args.delta_time,
        min_green=10,
        max_green=50,
        reward_fn=gangnam_reward,
        observation_class=GangnamObservationFunction,
        single_agent=False,
        ts_ids=ts_ids,
        time_to_teleport=args.teleport
    )

    # Data storage: {agent_id: list of dicts}
    data = {ts_id: [] for ts_id in env.ts_ids}
    
    observations = env.reset()
    done = {"__all__": False}
    
    print(f"Starting data collection for {args.seconds} seconds...")
    step = 0
    
    try:
        while not done["__all__"]:
            actions = {}
            for ts_id in env.ts_ids:
                # Random policy for exploration
                actions[ts_id] = env.action_spaces(ts_id).sample()
                
            next_observations, rewards, done, info = env.step(actions)
            
            for ts_id in env.ts_ids:
                # Store transition: s, a, r, s_next
                # State is [queue, arrival, neighbor_pressure]
                obs = observations[ts_id]
                next_obs = next_observations[ts_id]
                
                transition = {
                    "q": obs[0],
                    "lam": obs[1],
                    "neigh": obs[2],
                    "action_idx": actions[ts_id],
                    "reward": rewards[ts_id],
                    "q_next": next_obs[0],
                    "lam_next": next_obs[1],
                    "neigh_next": next_obs[2]
                }
                data[ts_id].append(transition)
                
            observations = next_observations
            step += 1
            if step % 100 == 0:
                print(f"Step {step} completed.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Saving collected data...")
    finally:
        env.close()
        
        # Save to CSV
        for ts_id, transitions in data.items():
            if not transitions:
                continue
            df = pd.DataFrame(transitions)
            filename = f"{args.out_csv_name.replace('.csv', '')}_{ts_id}.csv"
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            df.to_csv(save_path, index=False)
            print(f"Saved data for {ts_id} to {save_path} ({len(df)} samples)")



