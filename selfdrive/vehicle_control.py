import threading
import time
from collections import namedtuple

Values = namedtuple("Values", ["steer", "throttle", "brake"])


class PhysicsControl:
    """
    Constructor
    """

    def __init__(self):
        # Wheel settings
        self._steer = 0  # between -1.0 1.0
        self._throttle = 0  # between 0 1.0
        self._brake_val = 0  # between 0 1.0
        self._maintain_time_stamp = 0
        self._next_interval = 1
        self._forward_speed = 0  # M / s
        self._current_speed = 0  # KM / h
        self._last_max = 0
        self._max_speed = 0

        # Flags
        self._turn = False
        self.steer_back_slowly = False

        # Targets
        self._throttle_target = 0
        self._steer_target = 0
        self._brake_target = 0

        # Threads
        threading.Thread(target=self._control_throttle_value).start()
        threading.Thread(target=self._control_steer_value).start()
        threading.Thread(target=self._brake_to_target_value).start()

    @property
    def steer(self):
        return self._steer

    @steer.setter
    def steer(self, value):
        if self.steer_back_slowly:
            if value > 0:
                self.right(value)
            else:
                self.left(abs(value))
        else:
            self._steer = value

    @property
    def brake_target(self):
        return self._brake_target

    @brake_target.setter
    def brake_target(self, value):
        self._brake_target = value


    @property
    def forward_speed(self):
        return self._forward_speed

    @forward_speed.setter
    def forward_speed(self, value):
        if value >= 0:
            self._forward_speed = value
            self._current_speed = value * 3.6
        else:
            raise ValueError(f"Forward speed of {value} cannot be negative")

    """
    Setters
    """
    def set_speed_values(self, forward_speed, max_speed):
        self.forward_speed = forward_speed
        self._current_speed = int(forward_speed * 3.6)
        self._max_speed = max_speed

    """
    Wheel functions
    * right
    * left
    """

    def right(self, value):
        self._turn = True
        if self._steer < value or not self.steer_back_slowly:
            self.steer_back_slowly = False
            self._steer_target = 0
            self._steer = value
        else:
            self._steer_target = value

    def left(self, value):
        self._turn = True
        if self._steer > value or not self.steer_back_slowly:
            self.steer_back_slowly = False
            self._steer_target = 0
            self._steer = value * -1
        else:
            self._steer_target = value * -1

    """
    Threads
    * control_steer_value
    * control_throttle_value
    * brake_to_target_value
    """

    def _control_steer_value(self):
        while True:
            if self._steer_target != 0:
                if self._steer < self._steer_target:
                    value = 0.1 if self._steer > 0 else 0.02
                    self._steer += value
                    if self._steer > self._steer_target:
                        self.steer_back_slowly = False
                        self._steer = self._steer_target
                else:
                    value = 0.1 if self._steer < 0 else 0.02
                    self._steer -= value
                    if self._steer < self._steer_target:
                        self.steer_back_slowly = False
                        self._steer = self._steer_target
                time.sleep(0.2)
            else:
                time.sleep(0.1)

    def _control_throttle_value(self):
        while True:
            if self._throttle < self._throttle_target:
                self._throttle += 0.075
                if self._throttle > self._throttle_target:
                    self._throttle = self._throttle_target
            time.sleep(0.25)

    def brake(self, value):
        self._brake_val = value
        self._last_max = 0
        self._throttle = 0
        self._throttle_target = 0

    def accelerate(self, value):
        self._throttle_target = value
        self._last_max = 0

    def set_brake_target(self, speed):
        self._brake_target = speed

    def _brake_to_target_value(self):
        while True:
            if self._brake_target != 0 and self._brake_target < self._current_speed:
                diff = self._current_speed - self._brake_target
                if diff > 20:
                    value = 0.5
                elif diff > 15:
                    value = 0.25
                elif diff > 10:
                    value = 0.15
                elif diff > 5:
                    value = diff / 100
                else:
                    value = diff / 200
                if self._brake_target < 15:
                    value *= 1.5
                self.brake(value)
            time.sleep(0.2)

    def set_maintain_settings(self, max_speed=None):
        if self._last_max == 0 and self.forward_speed > 0.1:
            if not max_speed:
                speed = self._current_speed
            else:
                speed = max_speed
            if speed > self._max_speed:
                speed = self._max_speed
            self._last_max = speed
            if speed <= 12:
                self._throttle = 0.45
            if speed <= 16:
                self._throttle = 0.5
            elif speed <= 22:
                self._throttle = 0.55
            elif speed <= 35:
                self._throttle = 0.6
            else:
                self._throttle = 0.7

    def maintain_speed(self):
        if time.time() - self._maintain_time_stamp > self._next_interval and self._last_max > 0:
            if abs(self._current_speed - self._last_max) > 20:
                self.brake(0.3)
                self._next_interval = 1
            elif abs(self._current_speed - self._last_max) > 10:
                if self._current_speed > self._last_max:
                    self._throttle -= 0.07 if self._last_max > 15 else 0.05
                else:
                    self._throttle += 0.07 if self._last_max > 15 else 0.05
                self._next_interval = 3 if self._last_max > 15 else 1
            elif abs(self._current_speed - self._last_max) > 5:
                if self._current_speed > self._last_max:
                    self._throttle -= 0.04
                else:
                    self._throttle += 0.04
                self._next_interval = 3 if self._last_max > 15 else 1

            self._maintain_time_stamp = time.time()

    def reset_brake_value(self):
        self._brake_val = 0

    def has_values(self):
        return self._steer != 0 or self._throttle != 0 or self.brake != 0

    def get_values(self):
        return Values(steer=self._steer, throttle=self._throttle, brake=self._brake_val)
