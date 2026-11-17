"""Batched quaternion and angle helpers.

World frame: x forward (north), y left (west), z up. Body frame: x forward, y left, z up.
Quaternions are stored scalar-first, shape (count, 4).
"""
import numpy as np

CONJUGATE_SIGNS = np.array([1, -1, -1, -1])


def rotate_body_to_world(attitude, vectors):
    scalar = attitude[:, 0:1]
    imaginary = attitude[:, 1:]
    twice_cross = 2.0 * np.cross(imaginary, vectors)
    return vectors + scalar * twice_cross + np.cross(imaginary, twice_cross)


def rotate_world_to_body(attitude, vectors):
    conjugate = attitude * CONJUGATE_SIGNS.astype(attitude.dtype)
    return rotate_body_to_world(conjugate, vectors)


def quaternion_from_euler(roll, pitch, yaw):
    cos_roll, sin_roll = np.cos(roll / 2), np.sin(roll / 2)
    cos_pitch, sin_pitch = np.cos(pitch / 2), np.sin(pitch / 2)
    cos_yaw, sin_yaw = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
                     sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
                     cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
                     cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw], axis=1)


def euler_from_quaternion(attitude):
    """Returns (roll, pitch, yaw) in radians."""
    qw, qx, qy, qz = attitude[:, 0], attitude[:, 1], attitude[:, 2], attitude[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def heading_of(attitude):
    return euler_from_quaternion(attitude)[2]


def wrap_angle(angle):
    """Angle -> [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def upright_cosine(attitude):
    """Cosine of the tilt: 1 level, 0 on its side, negative upside down."""
    return 1 - 2 * (attitude[:, 1] ** 2 + attitude[:, 2] ** 2)


def tilt_radians(attitude):
    return np.arccos(np.clip(upright_cosine(attitude), -1, 1))


def tilt_degrees(attitude):
    return np.rad2deg(tilt_radians(attitude))


def quaternion_derivative(attitude, body_rates):
    qw, qx, qy, qz = attitude[:, 0], attitude[:, 1], attitude[:, 2], attitude[:, 3]
    rate_x, rate_y, rate_z = body_rates[:, 0], body_rates[:, 1], body_rates[:, 2]
    return 0.5 * np.stack([
        -qx * rate_x - qy * rate_y - qz * rate_z,
        qw * rate_x + qy * rate_z - qz * rate_y,
        qw * rate_y - qx * rate_z + qz * rate_x,
        qw * rate_z + qx * rate_y - qy * rate_x], 1)
