import argparse
import os
import sys
# Add project root to sys.path to allow running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import joblib

# ============================================
# ST-FQI Agent 
# ============================================

class STFQI_Agent:
    def __init__(
        self,
        tau_support: float = 0.05,
        gamma: float = 0.95,
        n_iters: int = 15
    ):
        self.tau = tau_support
        self.gamma = gamma
        self.n_iters = n_iters

        self.behavior_clf = RandomForestClassifier(
            n_estimators= 150,
            max_depth= None,
            min_samples_leaf=2,
            
            n_jobs=-1
        )
        self.q_reg = ExtraTreesRegressor(
            n_estimators=150,
            max_depth= None,
            min_samples_leaf= 2,
            n_jobs=-1
        )
        self.is_fitted = False

    @staticmethod
    def _build_features_state(s: np.ndarray) -> np.ndarray:
        """
        s: (N, 3) = [q, lam, neigh]
        """
        return s

    @staticmethod
    def _build_features_state_action(s: np.ndarray, a_idx: np.ndarray, n_actions: int) -> np.ndarray:
        """
        Concatenate state with one-hot action.
        """
        N = s.shape[0]
        a_onehot = np.zeros((N, n_actions), dtype=float)
        a_onehot[np.arange(N), a_idx] = 1.0
        return np.concatenate([s, a_onehot], axis=1)

    def fit_behavior_model(self, s_arr: np.ndarray, a_arr: np.ndarray):
        X = self._build_features_state(s_arr)
        y = a_arr
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
        self.behavior_clf.fit(X_train, y_train)
        y_val_proba = self.behavior_clf.predict_proba(X_val)
        try:
            ll = log_loss(y_val, y_val_proba)
            print(f"[ST-FQI] Behavior classifier log-loss: {ll:.3f}")
        except ValueError:
            print("[ST-FQI] Behavior classifier log-loss: N/A (single class)")

    def _support_actions(self, s: np.ndarray) -> np.ndarray:
        proba = self.behavior_clf.predict_proba(self._build_features_state(s))
        support_mask = proba >= self.tau
        return support_mask, proba

    def fit_fqi(self, dataset: pd.DataFrame, n_actions: int):
        """
        dataset columns:
          q, lam, neigh, action_idx, reward, q_next, lam_next, neigh_next
        """
        # Prepare arrays
        s = dataset[["q", "lam", "neigh"]].values.astype(float)
        a = dataset["action_idx"].values.astype(int)
        r = dataset["reward"].values.astype(float)
        s_next = dataset[["q_next", "lam_next", "neigh_next"]].values.astype(float)

        # 1) fit behavior model
        print("Fitting behavior model...")
        self.fit_behavior_model(s, a)

        # 2) initialize Q with zeros (via supervised regression on r only or small rollout)
        N = s.shape[0]
        X_sa = self._build_features_state_action(s, a, n_actions)
        y = r.copy()  # first iteration: 1-step reward only
        print("Initializing Q-function...")
        self.q_reg.fit(X_sa, y)

        # 3) FQI iterations with support-aware targets
        for it in range(self.n_iters):
            # Predict current Q for all actions at s_next
            support_mask, proba = self._support_actions(s_next)
            # For each s_next, choose best supported action
            n_samples = s_next.shape[0]
            target = np.zeros(n_samples, dtype=float)

            # Precompute Q(s_next, a) for all actions by stacking
            all_q_vals = []
            for a_idx in range(n_actions):
                a_vec = np.full(n_samples, a_idx, dtype=int)
                X_next = self._build_features_state_action(s_next, a_vec, n_actions)
                q_vals = self.q_reg.predict(X_next)
                all_q_vals.append(q_vals)
            all_q_vals = np.stack(all_q_vals, axis=1)  # (N, n_actions)

            for i in range(n_samples):
                supported = np.where(support_mask[i])[0]
                if len(supported) > 0:
                    best_idx = supported[np.argmax(all_q_vals[i, supported])]
                else:
                    # Fallback to behavior argmax
                    best_idx = int(np.argmax(proba[i]))
                target[i] = r[i] + self.gamma * all_q_vals[i, best_idx]

            # Refit regressor on (s,a) -> target
            X_sa = self._build_features_state_action(s, a, n_actions)
            self.q_reg.fit(X_sa, target)
            print(f"[ST-FQI] Iter {it+1}/{self.n_iters} done.")

        self.is_fitted = True

# ============================================
# Training Script
# ============================================

if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="Train ST-FQI Agents for Gangnam")
    prs.add_argument("-data_prefix", type=str, default="gangnam_data", help="Prefix of data CSVs")
    prs.add_argument("-out_dir", type=str, default="models", help="Directory to save models")
    prs.add_argument("-net", dest="net_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int2.net.xml")
    prs.add_argument("-route", dest="route_file", type=str, default="/Users/kierankhan/Dev/sumo-rl/sumo_rl/nets/gangnam/gangnam_int_scaled.rou.xml")
    prs.add_argument("--ts_id", dest="ts_id", type=str, default=None, help="Specific Traffic Signal ID to train (e.g. J0). If None, trains all.")
    args = prs.parse_args()

    # Find data files
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = [f for f in os.listdir(data_dir) if f.startswith("gangnam_data_") and f.endswith(".csv")]
    
    if args.ts_id:
        data_files = [f for f in data_files if f"_{args.ts_id}.csv" in f]
        if not data_files:
            print(f"No data files found for agent {args.ts_id}")
            sys.exit(1)

    if not data_files:
        print(f"No data files found in {data_dir}!")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize env to get action spaces
    print("Initializing environment to determine action spaces...")
    from sumo_rl import SumoEnvironment
    from sumo_rl.gangnam_utils import GangnamObservationFunction, gangnam_reward
    
    ts_ids = [args.ts_id] if args.ts_id else None
    
    env = SumoEnvironment(
        net_file=args.net_file,
        route_file=args.route_file,
        use_gui=False,
        num_seconds=10000, # Short, just to init
        reward_fn=gangnam_reward,
        observation_class=GangnamObservationFunction,
        single_agent=True,
        ts_ids=ts_ids
    )

    for f in data_files:
        ts_id = f.replace(args.data_prefix + "_", "").replace(".csv", "")
        
        if ts_id not in env.ts_ids:
            print(f"Skipping {ts_id} (not in environment)")
            continue
            
        n_actions = env.action_spaces(ts_id).n
        print(f"\nTraining agent for {ts_id} using {f} (n_actions={n_actions})...")
        
        df = pd.read_csv(os.path.join(data_dir, f))
        agent = STFQI_Agent(n_iters=20)
        
        # Store n_actions in agent for later use
        agent.n_actions = n_actions
        
        agent.fit_fqi(df, n_actions=n_actions)
        
        # Save model
        save_path = os.path.join(args.out_dir, f"fqi_model_{ts_id}.joblib")
        joblib.dump(agent, save_path)
        print(f"Model saved to {save_path}")
    
    env.close()
