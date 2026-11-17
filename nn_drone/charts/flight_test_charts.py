"""Charts for the flight tests."""
import numpy as np

from nn_drone.charts.style import (BACKGROUND, INK, PWM_AXIS_US, SECONDARY, SERIES, figure, plot_motor_pwm, save)
from nn_drone.flight_tests.long_route import CIRCLE_LAP, WAYPOINTS

SAMPLES_PER_MILLION = 1e6
WAYPOINT_LABEL_OFFSET = (8, 8)
SECONDS_PER_MINUTE = 60


def waypoint_mission_chart(path, recording, mission, training_log):
    mission = np.asarray(mission)
    time, position = recording["time"], recording["position"]
    fig, axes = figure(5, 17, [1.5, 1, 1, 1, 1])
    top = axes[0]
    top.plot(mission[:, 0], mission[:, 1], "--", color=SECONDARY, lw=1, label="Planned path")
    top.plot(position[:, 0], position[:, 1], color=SERIES[0], lw=2, label="Flown path")
    top.scatter(mission[:, 0], mission[:, 1], s=60, color=BACKGROUND, edgecolor=SECONDARY, zorder=3)
    for number, waypoint in enumerate(mission[:-1], start=1):
        top.annotate(f"WP{number}", (waypoint[0], waypoint[1]), textcoords="offset points", xytext=WAYPOINT_LABEL_OFFSET,
                     color=SECONDARY)
    top.set_aspect("equal")
    top.set_xlabel("x forward (m)")
    top.set_ylabel("y left (m)")
    top.set_title("Waypoint mission, top view")
    top.legend(frameon=False, loc="center")
    _height_panel(axes[1], time, recording["target"][:, 2], position[:, 2], "Altitude above start (m)")
    attitude = np.rad2deg(recording["roll_pitch_yaw"])
    axes[2].plot(time, attitude[:, 0], color=SERIES[0], lw=2, label="Roll")
    axes[2].plot(time, attitude[:, 1], color=SERIES[1], lw=2, label="Pitch")
    axes[2].set_title("Roll and pitch (degrees)")
    axes[2].legend(frameon=False, ncol=2)
    plot_motor_pwm(axes[3], time, recording["pwm"])
    axes[3].set_ylim(*PWM_AXIS_US)
    axes[3].set_title("Network output: motor PWM (µs)")
    axes[3].set_xlabel("Time (s)")
    samples = np.array([entry["samples"] for entry in training_log]) / SAMPLES_PER_MILLION
    axes[4].plot(samples, [entry["crash_per_1k"] for entry in training_log], color=SERIES[0], lw=2)
    axes[4].set_title("Training: crashes per 1,000 control steps")
    axes[4].set_xlabel("Training samples (millions)")
    save(fig, path)


def _height_panel(axis, time, target, actual, title, legend_location="best"):
    axis.plot(time, target, "--", color=SECONDARY, lw=1, label="Target")
    axis.plot(time, actual, color=SERIES[0], lw=2, label="Actual")
    axis.set_title(title)
    axis.legend(frameon=False, loc=legend_location)


def circle_chart(path, plan, nominal):
    recording = nominal.recording
    time, position = recording["time"], recording["position"]
    fig, axes = figure(4, 16, [2.2, 1, 1, 1])
    top = axes[0]
    angles = np.linspace(0, 2 * np.pi, 200)
    top.plot(plan.radius_m * np.cos(angles), plan.radius_m * np.sin(angles), "--", color=SECONDARY, lw=1, label="Planned circle")
    top.plot(position[:, 0], position[:, 1], color=SERIES[0], lw=2, label="Flown path")
    top.scatter([0], [0], s=50, color=BACKGROUND, edgecolor=SECONDARY, zorder=3)
    top.annotate("Start", (0, 0), textcoords="offset points", xytext=(-10, 10), color=SECONDARY)
    top.set_aspect("equal")
    top.set_xlabel("x forward (m)")
    top.set_ylabel("y left (m)")
    top.set_title(f"Circle, top view (radius {plan.radius_m:.0f} m, {plan.speed_mps:g} m/s, {plan.mass_kg:g} kg, "
                  f"{plan.altitude_asl_m:g} m)")
    top.legend(frameon=False, loc="lower left")
    radial = axes[1]
    radial.plot(time, nominal.radial_error_trace, color=SERIES[0], lw=2)
    radial.axvspan(plan.approach_s, plan.approach_s + plan.lap_s, color="#e8e7e2", alpha=0.5, lw=0)
    first_lap_step = int(plan.approach_s / float(time[1] - time[0]))
    limit = max(1.0, 1.2 * np.abs(nominal.radial_error_trace[first_lap_step:]).max())
    radial.set_ylim(-limit, limit)
    radial.set_title("Distance from the circle line (m), shaded = lap; + is outside")
    _height_panel(axes[2], time, plan.start_altitude_asl_m + recording["target"][:, 2],
                  plan.start_altitude_asl_m + position[:, 2], "Altitude above sea level (m)", "lower right")
    axes[2].ticklabel_format(useOffset=False)
    plot_motor_pwm(axes[3], time, recording["pwm"])
    axes[3].set_ylim(*PWM_AXIS_US)
    axes[3].set_title("Network output: motor PWM (µs)")
    axes[3].set_xlabel("Time (s)")
    save(fig, path)


def envelope_chart(path, climb_recording, sweep_rows):
    time, height = climb_recording["time"], climb_recording["position"][:, 2]
    mean_pwm = climb_recording["pwm"].mean(1)
    fig, axes = figure(4, 14)
    _height_panel(axes[0], time, climb_recording["target"][:, 2], height, "Climb to 2000 m: altitude (m)", "lower right")
    axes[0].set_xlabel("Time (s)")
    axes[1].plot(height, mean_pwm, color=SERIES[0], lw=1.5)
    axes[1].set_ylim(*PWM_AXIS_US)
    axes[1].set_title("Average motor PWM (µs) against altitude: thinner air needs more throttle")
    axes[1].set_xlabel("Altitude (m)")
    for axis, key, color, title in ((axes[2], "completed", SERIES[0], "share of 200 randomised drones completing the square mission"),
                                    (axes[3], "crashed", SERIES[1], "share that crashed")):
        _share_bars(axis, sweep_rows, key, color, f"Mass sweep: {title}")
    save(fig, path)


def _share_bars(axis, rows, key, color, title):
    positions = np.arange(len(rows))
    shares = [100 * row[key] / row["n"] for row in rows]
    axis.bar(positions, shares, width=0.6, color=color)
    for position, row, share in zip(positions, rows, shares):
        axis.text(position, share + 2, f"{100 * row[key] // row['n']}%", ha="center", color=INK)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{row['mass_kg']} kg" for row in rows])
    axis.set_ylim(0, 115)
    axis.grid(axis="x", visible=False)
    axis.set_title(title)


def _route_side_and_top_views(axes, target, position):
    side, top = axes
    side.plot(target[:, 1], target[:, 2], "--", color=SECONDARY, lw=1, label="Planned")
    side.plot(position[:, 1], position[:, 2], color=SERIES[0], lw=2, label="Flown")
    for name, waypoint in WAYPOINTS.items():
        side.scatter([waypoint[1]], [waypoint[2]], s=45, color=BACKGROUND, edgecolor=SECONDARY, zorder=3)
        side.annotate(name, (waypoint[1], waypoint[2]), textcoords="offset points", xytext=(8, 6), color=SECONDARY)
    sideways_label = "← left     sideways position (m)     right →"
    side.invert_xaxis()
    side.set_xlabel(sideways_label)
    side.set_ylabel("height (m)")
    side.set_title("Side view: height against sideways position")
    side.legend(frameon=False, loc="center left")
    top.plot(target[:, 1], target[:, 0], "--", color=SECONDARY, lw=1, label="Planned")
    top.plot(position[:, 1], position[:, 0], color=SERIES[0], lw=2, label="Flown")
    top.invert_xaxis()
    top.set_aspect("equal")
    top.set_xlabel(sideways_label)
    top.set_ylabel("forward (m)")
    top.set_title("Top view: circle at WP5 (left) and rectangle at WP6 (right)")
    top.legend(frameon=False, loc="lower center", ncol=2)


def _pwm_window(axis, time, pwm, start, end, title):
    window = (time >= start) & (time <= end)
    plot_motor_pwm(axis, time[window] - start, pwm[window])
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    low, high = pwm[window].min(), pwm[window].max()
    axis.set_ylim(max(PWM_AXIS_US[0], low - 0.35 * (high - low) - 20), min(PWM_AXIS_US[1], high + 20))


def long_mission_chart(path, nominal):
    recording = nominal.recording
    minutes, position, target = recording["time"] / SECONDS_PER_MINUTE, recording["position"], recording["target"]
    fig, axes = figure(7, 29, [1.6, 1.2, 1, 1, 1.1, 1.1, 1.1])
    _route_side_and_top_views(axes[:2], target, position)
    _height_panel(axes[2], minutes, target[:, 2], position[:, 2], "Height (m)", "upper left")
    axes[3].plot(minutes, np.linalg.norm(target - position, axis=1), color=SERIES[0], lw=1.5)
    axes[3].set_title("Distance from the moving target point (m)")
    plot_motor_pwm(axes[4], minutes, recording["pwm"], line_width=0.8)
    axes[4].set_ylim(*PWM_AXIS_US)
    axes[4].set_title("Actual PWM sent to each motor (µs), whole mission")
    axes[4].set_xlabel("Time (minutes)")
    pwm, period = nominal.pwm_every_step, nominal.period
    time = np.arange(1, len(pwm) + 1) * period
    _pwm_window(axes[5], time, pwm, 0, 12, "Actual PWM, zoom: first 12 s (take-off)")
    lap_start = nominal.route.steps_in(CIRCLE_LAP)[0] * period
    _pwm_window(axes[6], time, pwm, lap_start - 3, lap_start + 12, "Actual PWM, zoom: start of the circle lap")
    save(fig, path)


def full_mission_chart(path, nominal, timing):
    recording = nominal.recording
    minutes, position, target = recording["time"] / SECONDS_PER_MINUTE, recording["position"], recording["target"]
    pwm, period = nominal.pwm_every_step, nominal.period
    time = np.arange(1, len(pwm) + 1) * period
    fig, axes = figure(6, 28, [1.6, 1.2, 1.3, 1.3, 1, 1])
    _route_side_and_top_views(axes[:2], target, position)
    armed, spool = timing.disarmed_s, timing.disarmed_s + timing.armed_idle_s
    network = spool + timing.spool_up_s
    _marked_pwm_panel(axes[2], time, pwm, 0, 14, "Start: disarmed to take-off, actual PWM per motor (µs)",
                      [(0, "disarmed"), (armed, "armed, idle"), (spool, "spool-up"), (network, "network in control"),
                       (nominal.lift_off_time[0], "lift-off")])
    touchdown, detected, disarmed = nominal.touchdown_time[0], nominal.detection_time[0], nominal.disarm_time[0]
    _marked_pwm_panel(axes[3], time, pwm, touchdown - 6, disarmed + 2, "End: landing to disarm, actual PWM per motor (µs)",
                      [(touchdown, "touchdown"), (detected, "landing detected"), (detected + timing.ramp_to_idle_s, "idle"),
                       (disarmed, "disarmed")])
    plot_motor_pwm(axes[4], minutes, recording["pwm"], line_width=0.8)
    axes[4].set_ylim(950, 1900)
    axes[4].set_title("Whole mission: actual PWM per motor (µs)")
    axes[4].set_xlabel("Time (minutes)")
    _height_panel(axes[5], minutes, target[:, 2], position[:, 2], "Height (m)", "upper left")
    axes[5].set_xlabel("Time (minutes)")
    save(fig, path)


def _marked_pwm_panel(axis, time, pwm, start, end, title, marks):
    window = (time >= start) & (time <= end)
    plot_motor_pwm(axis, time[window] - start, pwm[window], line_width=1.3)
    axis.set_ylim(950, 1900)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    for moment, label in marks:
        axis.axvline(moment - start, color=SECONDARY, lw=0.8, ls=":")
        axis.text(moment - start + 0.08, 1880, label, rotation=90, va="top", ha="left", color=SECONDARY, fontsize=9)
