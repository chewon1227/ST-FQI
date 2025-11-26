import argparse
import os
import sys
# Add project root to sys.path to allow running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from sumo_rl import SumoEnvironment
from sumo_rl.gangnam_utils import GangnamObservationFunction, gangnam_reward
from sumo_rl.train_fqi import STFQI_Agent # Need class definition for joblib load

def evaluate(net_file, route_file, model_dir, seconds, delta_time=60, out_csv_name="gangnam_eval", ts_id=None, time_to_teleport=-1):
    print("Initializing environment for evaluation...")
    
    ts_ids = [ts_id] if ts_id else None
    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        reward_fn=gangnam_reward,
        observation_class=GangnamObservationFunction,
        single_agent=False,
        ts_ids=ts_ids,
        time_to_teleport=time_to_teleport
    )

    # Load agents
    agents = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, "..", model_dir) # Assuming run from root
    if not os.path.exists(model_dir_path):
         model_dir_path = os.path.join(script_dir, model_dir) # Try local

    print(f"Loading models from {model_dir_path}...")
    
    for ts_id in env.ts_ids:
        model_path = os.path.join(model_dir_path, f"fqi_model_{ts_id}.joblib")
        if os.path.exists(model_path):
            agents[ts_id] = joblib.load(model_path)
            print(f"Loaded model for {ts_id}")
        else:
            print(f"WARNING: No model found for {ts_id}, will use random actions.")
            agents[ts_id] = None

    observations = env.reset()
    done = {"__all__": False}
    
    print(f"Starting evaluation for {seconds} seconds...")
    step = 0
    total_rewards = {ts: 0.0 for ts in env.ts_ids}
    
    while not done["__all__"] and step < seconds:
        actions = {}
        for ts_id in env.ts_ids:
            agent = agents.get(ts_id)
            if agent:
                # Prepare state for prediction
                # Agent expects (N, 3) array
                obs = observations[ts_id]
                s_vec = np.array([obs]) # Shape (1, 3)
                
                # We need to implement act_greedy or similar in STFQI_Agent or do it here
                # The STFQI_Agent in train_fqi.py doesn't have a clean 'act' method for single state
                # Let's do it manually here using the logic from train_fqi
                
                # 1. Check support
                support_mask, proba = agent._support_actions(s_vec)
                support_mask = support_mask[0]
                proba = proba[0]
                
                n_actions = env.action_spaces(ts_id).n
                
                # 2. Predict Q for all actions
                all_q = []
                for a_idx in range(n_actions):
                    a_vec = np.array([a_idx])
                    X_sa = agent._build_features_state_action(s_vec, a_vec, n_actions)
                    q_val = agent.q_reg.predict(X_sa)[0]
                    all_q.append(q_val)
                all_q = np.array(all_q)
                
                # 3. Choose action
                supported = np.where(support_mask)[0]
                if len(supported) > 0:
                    best_idx = supported[np.argmax(all_q[supported])]
                else:
                    best_idx = np.argmax(proba)
                
                actions[ts_id] = int(best_idx)
            else:
                actions[ts_id] = env.action_spaces(ts_id).sample()
            
        next_observations, rewards, done, info = env.step(actions)
        
        for ts_id in env.ts_ids:
            total_rewards[ts_id] += rewards[ts_id]
            
        observations = next_observations
        step += 1
        if step % 100 == 0:
            print(f"Step {step} completed.")

    env.close()
    print("Evaluation finished.")
    print("Total Rewards:", total_rewards)
    
    # Save metrics
    env.save_csv(out_csv_name, 1)
    print(f"Saved metrics to {out_csv_name}")

if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="Evaluate ST-FQI Agents")
    prs.add_argument("-net", dest="net_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int.net.xml")
    prs.add_argument("-route", dest="route_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int.rou.xml")
    prs.add_argument("-seconds", dest="seconds", type=int, default=86400)
    prs.add_argument("-delta", dest="delta_time", type=int, default=60, help="Delta time (cycle length)")
    prs.add_argument("-models", dest="model_dir", type=str, default="models")
    prs.add_argument("--baseline", action="store_true", help="Run with random baseline policy")
    prs.add_argument("--out", dest="out_csv_name", type=str, default="gangnam_eval")
    prs.add_argument("--ts_id", dest="ts_id", type=str, default=None, help="Specific Traffic Signal ID to evaluate (e.g. J0). If None, evaluates all.")
    prs.add_argument("--teleport", dest="teleport", type=int, default=-1, help="Time to teleport stuck vehicles (seconds). -1 to disable.\n")
    
    args = prs.parse_args()
    
    # If baseline, force model_dir to non-existent to trigger random policy
    model_dir = "non_existent_dir" if args.baseline else args.model_dir
    out_name = args.out_csv_name + "_baseline" if args.baseline else args.out_csv_name
    
    evaluate(args.net_file, args.route_file, model_dir, args.seconds, delta_time=args.delta_time, out_csv_name=out_name, ts_id=args.ts_id, time_to_teleport=args.teleport)
