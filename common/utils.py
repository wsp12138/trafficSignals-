"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from sumo_rl.environment.traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        # lane_wait_time = self.ts.get_accumulated_waiting_time_per_lane()[:12]
        # lane_queue_length = {t: [self.env.sumo_env.sumo.lane.getLastStepHaltingNumber(lane) for lane in
        #                          self.env.sumo_env.traffic_signals[t].lanes][:12] for t in self.ts}
        # wait_person_time, wait_person_number = self.get_person_wait_time_and_number_()

        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        # print(phase_id,'phase_id')
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        # return spaces.Box(low=np.zeros(2 * len(self.ts.lanes) - 8 + 2, dtype=np.float32),
        #                   high=np.full(2 * len(self.ts.lanes) - 8 + 2, np.inf, dtype=np.float32))

        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
