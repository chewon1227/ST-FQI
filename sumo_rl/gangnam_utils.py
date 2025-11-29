import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

class GangnamObservationFunction(ObservationFunction):
    """
    Custom Observation Function to match the reference ST-FQI implementation.
    State: [queue_length, arrival_proxy, neighbor_pressure]
    """
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        # 1. Queue (Total halting vehicles)
        queue = float(self.ts.get_total_queued())

        # 2. Arrivals (Total vehicles on incoming lanes as a proxy for volume)
        # Note: This is a snapshot count, not a rate over time, but serves as a state feature.
        arrivals = sum(self.ts.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.ts.lanes)

        # 3. Neighbor Pressure
        # neighbor_pressure = sum(q_j) - q_i
        # We need to access other agents.
        # self.ts.env.traffic_signals is a dict {ts_id: TrafficSignal}
        total_neighbor_queue = 0.0
        if self.ts.env and self.ts.env.traffic_signals:
            for ts_id, other_ts in self.ts.env.traffic_signals.items():
                if ts_id != self.ts.id:
                    total_neighbor_queue += float(other_ts.get_total_queued())
        
        neighbor_pressure = total_neighbor_queue - queue

        return np.array([queue, arrivals, neighbor_pressure], dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        # Shape is 3: [queue, arrivals, neighbor_pressure]
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,), 
            dtype=np.float32
        )

def gangnam_reward(ts: TrafficSignal):
    """
    Custom Reward Function: -queue * cycle_length
    """
    return -float(ts.get_total_queued()) * ts.delta_time
