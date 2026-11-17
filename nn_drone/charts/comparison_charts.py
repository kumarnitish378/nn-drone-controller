"""Charts for the ArduPilot comparison, the stability checks and the real-log replay."""
import numpy as np

from nn_drone.charts.style import COMPARISON, GRID, SECONDARY, SERIES, figure, motor_legend_columns, save

HOVER_ZOOM_S = 10
MIN_HALF_RANGE_US = 60
RANGE_PADDING_US = 15
LEGEND_SPACE_US = 25
SHADOW_WINDOW_S = 40
GUST_PRE_ROLL_S = 5
CLIMB_OUT_VIEW_S = (2, 14)
BAR_WIDTH = 0.38
ARDUPILOT_COLOR, NETWORK_COLOR = COMPARISON[0], COMPARISON[1]


def comparison_chart(path, flights, traces, planned_route):
    rows = 4 + len(flights)
    fig, axes = figure(rows, 4.6 + 3.3 * (rows - 1), [1.6] + [1] * (rows - 1))
    top = axes[0]
    top.plot(planned_route[:, 1], planned_route[:, 0], "--", color=SECONDARY, lw=1, label="Planned route")
    for flight, color in zip(flights, COMPARISON):
        top.plot(flight.east, flight.north, color=color, lw=1.6, label=flight.name)
    top.set_aspect("equal")
    top.set_xlabel("East (m)")
    top.set_ylabel("North (m)")
    top.set_title("Top view")
    top.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
    for flight, trace, color in zip(flights, traces, COMPARISON):
        axes[1].plot(flight.time, flight.height, color=color, lw=1.6, label=flight.name)
        axes[2].plot(flight.time, trace["route_error"], color=color, lw=1.2, label=flight.name)
        pwm_time, pwm = trace["pwm_10hz"]
        axes[3].plot(pwm_time, pwm.mean(1), color=color, lw=1.2, label=flight.name)
    axes[1].set_title("Height (m)")
    axes[1].legend(frameon=False, loc="lower center", ncol=1)
    axes[2].set_title("Distance from the planned route (m)")
    axes[2].set_ylim(0, None)
    axes[2].legend(frameon=False, loc="upper right")
    axes[3].set_title("Average motor PWM, whole flight (µs)")
    axes[3].set_ylim(1000, 2000)
    axes[3].legend(frameon=False, loc="lower center", ncol=1)
    for axis, flight, trace in zip(axes[4:], flights, traces):
        pwm_time, pwm = trace["pwm_10hz"]
        hover_start = trace["hover_window"][0]
        window = (pwm_time >= hover_start) & (pwm_time <= hover_start + HOVER_ZOOM_S)
        for motor in range(pwm.shape[1]):
            axis.plot(pwm_time[window] - hover_start, pwm[window, motor], color=SERIES[motor], lw=1.3, label=f"Motor {motor + 1}")
        low, high = pwm[window].min(), pwm[window].max()
        middle, half = (low + high) / 2, max(MIN_HALF_RANGE_US, (high - low) / 2 + RANGE_PADDING_US)
        axis.set_ylim(middle - half - LEGEND_SPACE_US, middle + half)
        axis.set_title(f"{flight.name}: each motor's PWM during 10 s of the hover (µs)")
        axis.legend(frameon=False, ncol=motor_legend_columns(pwm.shape[1]), loc="lower center")
    axes[-1].set_xlabel("Time (s)")
    save(fig, path, vertical_padding=1.6)


def stability_chart(path, gusts, attitudes, gust_duration_s, after_gust_s):
    fig, axes = figure(3, 11)
    for (name, gust), color in zip(gusts.items(), COMPARISON):
        time, displacement, tilt = gust["_trace"]
        window = (time > -GUST_PRE_ROLL_S) & (time < gust_duration_s + after_gust_s)
        axes[0].plot(time[window], displacement[window], color=color, lw=1.6, label=name)
        axes[1].plot(time[window], tilt[window], color=color, lw=1.3, label=name)
    for axis in axes[:2]:
        axis.axvspan(0, gust_duration_s, color=GRID, alpha=0.6, lw=0)
        axis.legend(frameon=False, loc="upper right")
    axes[0].set_title("Gust test: distance pushed off the hover point (m), shaded = 8 m/s wind")
    axes[1].set_title("Gust test: tilt (degrees)")
    axes[1].set_xlabel("Time from gust start (s)")
    for (name, attitude), color in zip(attitudes.items(), COMPARISON):
        window = (attitude[:, 0] > CLIMB_OUT_VIEW_S[0]) & (attitude[:, 0] < CLIMB_OUT_VIEW_S[1])
        heading = np.degrees(np.unwrap(np.radians(attitude[window, 3])))
        axes[2].plot(attitude[window, 0], heading - heading[0], color=color, lw=1.5, label=name)
    axes[2].set_title("Mission log: heading change during climb-out (degrees)")
    axes[2].set_xlabel("Time from arming (s)")
    axes[2].legend(frameon=False, loc="upper right")
    save(fig, path, vertical_padding=1.6)


def real_log_chart(path, samples, stretches, stretch_stats, tracks):
    from nn_drone.ardupilot_comparison.real_log_replay import SAMPLE_COLUMNS, control_channels
    longest = int(np.argmax([end - start for start, end, _ in stretches]))
    auto_stretches = [index for index, (_, _, mode) in enumerate(stretches) if mode == "AUTO"]
    shown = auto_stretches[0] if auto_stretches else longest
    selected = samples[:, SAMPLE_COLUMNS["stretch"]] == shown
    time = samples[selected, 0] - samples[selected, 0][0]
    window = time < SHADOW_WINDOW_S
    ardupilot = control_channels(samples[:, SAMPLE_COLUMNS["ardupilot_pwm"]])
    network = control_channels(samples[:, SAMPLE_COLUMNS["network_pwm"]])
    mode = stretches[shown][2]
    fig, axes = figure(6, 21, [1, 1, 1, 1, 1.5, 1])
    titles = {"collective": "Collective (average of the four motors)", "roll": "Roll command", "pitch": "Pitch command",
              "yaw": "Yaw command"}
    for axis, (channel, title) in zip(axes[:4], titles.items()):
        real, shadow = ardupilot[channel][selected][window], network[channel][selected][window]
        if channel != "collective":
            real, shadow = real - real.mean(), shadow - shadow.mean()
        axis.plot(time[window], real, color=ARDUPILOT_COLOR, lw=1.4, label="ArduPilot, real flight")
        axis.plot(time[window], shadow, color=NETWORK_COLOR, lw=1.2, label="Network, shadow")
        axis.set_title(f"{title}, PWM µs ({mode} mode)")
        axis.legend(frameon=False, loc="upper right", ncol=2)
    axes[3].set_xlabel("Time in this stretch (s)")
    track, axis = tracks[shown], axes[4]
    axis.plot(track["target"][:, 1], track["target"][:, 0], "--", color=SECONDARY, lw=1, label="ArduPilot's target track")
    axis.plot(track["real"][:, 1], track["real"][:, 0], color=ARDUPILOT_COLOR, lw=1.5, label="Real flight (ArduPilot)")
    axis.plot(track["simulated"][:, 1], track["simulated"][:, 0], color=NETWORK_COLOR, lw=1.3, label="Network in our simulator")
    axis.set_aspect("equal")
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_title(f"Top view, {track['mode']} stretch: same target track")
    axis.legend(frameon=False, loc="best")
    axis = axes[5]
    positions = np.arange(len(stretch_stats))
    axis.bar(positions - BAR_WIDTH / 2, [stats["real_horizontal_error_mean_m"] for stats in stretch_stats], BAR_WIDTH,
             color=ARDUPILOT_COLOR, label="Real flight (ArduPilot)")
    axis.bar(positions + BAR_WIDTH / 2, [stats["network_sim_horizontal_error_mean_m"] for stats in stretch_stats], BAR_WIDTH,
             color=NETWORK_COLOR, label="Network in our simulator")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{stats['mode']}\n{stats['length_s']:.0f} s" for stats in stretch_stats])
    axis.grid(axis="x", visible=False)
    axis.set_title("Average horizontal distance from the target, each stretch (m)")
    axis.legend(frameon=False)
    save(fig, path, vertical_padding=1.5)
