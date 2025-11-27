import os
import sys
from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

env = SumoEnvironment(
    net_file="sumo_rl/nets/gangnam/gangnam_int.net.xml",
    route_file="sumo_rl/nets/gangnam/gangnam_int_scaled.rou.xml",
    use_gui=False,
    num_seconds=100,
    delta_time=5
)

print("TS IDs:", env.ts_ids)
for ts in env.ts_ids:
    print(f"\nTS: {ts}")
    print("Controlled Lanes:", env.traffic_signals[ts].lanes)
    print("Green Phase:", env.traffic_signals[ts].green_phase)

env.close()
