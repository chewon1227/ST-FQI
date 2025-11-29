import os
import sys
import traci
import sumolib

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

net_file = "sumo_rl/nets/gangnam/gangnam_int.net.xml"
route_file = "sumo_rl/nets/gangnam/gangnam_int_scaled.rou.xml"

sumo_cmd = [sumolib.checkBinary("sumo"), "-n", net_file, "-r", route_file, "--no-warnings"]
traci.start(sumo_cmd)

print("Running simulation to locate stopped vehicles...")
stopped_lanes = {}

for step in range(3600):
    traci.simulationStep()
    
    # Check every 100 steps
    if step % 100 == 0:
        pass

vehicles = traci.vehicle.getIDList()
vehicle_waits = []
for veh in vehicles:
    wait = traci.vehicle.getAccumulatedWaitingTime(veh)
    lane = traci.vehicle.getLaneID(veh)
    vehicle_waits.append((veh, lane, wait))

traci.close()

print("\nTop 10 Vehicles by Waiting Time:")
sorted_vehicles = sorted(vehicle_waits, key=lambda x: x[2], reverse=True)
for veh, lane, wait in sorted_vehicles[:10]:
    print(f"Veh: {veh}, Lane: {lane}, Wait: {wait:.2f}")

print(f"\nTotal System Waiting Time: {sum(v[2] for v in sorted_vehicles):.2f}")
