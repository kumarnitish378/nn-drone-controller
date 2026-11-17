"""Run the simulated vehicle in real time (or faster) with the MAVLink link attached."""
import time

from nn_drone.ground_link.mavlink_server import MavlinkServer
from nn_drone.ground_link.vehicle import SimulatedVehicle

MIN_SPEEDUP = 0.05
SLEEP_THRESHOLD_S = 0.002
MAX_SLEEP_S = 0.05
GIVE_UP_CATCHING_UP_S = 1.0


def run_ground_link(frame, weights_path, home, mass_kg, ground_station, speedup=1.0, duration_s=0.0, log=print):
    started = time.time()
    vehicle = SimulatedVehicle(frame, weights_path, home, mass_kg, log=log)
    vehicle.parameters["SIM_SPEEDUP"] = speedup
    server = MavlinkServer(vehicle, ground_station)
    vehicle.say("NN drone simulator (not ArduPilot)")
    vehicle.say(f"{frame} {mass_kg:g} kg ready")
    log(f"Sending MAVLink to {ground_station}. Open QGroundControl or Mission Planner (UDP 14550). Ctrl-C to stop.")
    real_start, simulated_start = time.time(), vehicle.time
    try:
        while not (duration_s and time.time() - started > duration_s):
            server.poll()
            vehicle.step()
            server.send_telemetry()
            speedup = max(float(vehicle.parameters["SIM_SPEEDUP"]), MIN_SPEEDUP)
            ahead_by = (vehicle.time - simulated_start) / speedup - (time.time() - real_start)
            if ahead_by > SLEEP_THRESHOLD_S:
                time.sleep(min(ahead_by, MAX_SLEEP_S))
            elif ahead_by < -GIVE_UP_CATCHING_UP_S:
                real_start, simulated_start = time.time(), vehicle.time
    except KeyboardInterrupt:
        pass
