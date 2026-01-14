# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

WHEEL_RADIUS = 0.055  # meters
BASE_WHEEL_DISTANCE = 0.16  # meters (half the distance between wheels)

def inv_kinematics(linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
    """Compute inverse kinematics for Upkie robot.

    Args:
        linear_velocity (float): Base linear velocity [m/s].
        angular_velocity (float): Base angular velocity [rad/s].

    Returns:
        tuple[float, float]: (Left wheel angular velocity, Right wheel angular velocity) [rad/s, rad/s].
    """
    w_left = (linear_velocity - BASE_WHEEL_DISTANCE * angular_velocity) / WHEEL_RADIUS
    w_right = -(linear_velocity + BASE_WHEEL_DISTANCE * angular_velocity) / WHEEL_RADIUS
    return w_left, w_right


def fw_kinematics(w_left: float, w_right: float) -> tuple[float, float]:
    """Compute forward kinematics for Upkie robot.

    Args:
        w_left (float): Left wheel angular velocity [rad/s].
        w_right (float): Right wheel angular velocity [rad/s].

    Returns:
        tuple[float, float]: (Base linear velocity, Base angular velocity) [m/s, rad/s].
    """
    linear_velocity = WHEEL_RADIUS * (w_left - w_right) / 2.0
    angular_velocity = -WHEEL_RADIUS * (w_left + w_right) / (2.0 * BASE_WHEEL_DISTANCE)
    return linear_velocity, angular_velocity