"""House style shared by every chart in the repository."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
SECONDARY = "#52514e"
BACKGROUND = "#fcfcfb"
GRID = "#e8e7e2"
AXIS_EDGE = "#c9c8c2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
COMPARISON = ("#eb6834", "#2a78d6", "#1baf7a")
FIGURE_WIDTH_IN = 7
DPI = 130
PWM_AXIS_US = (1000, 2000)
MOTORS_PER_LEGEND_ROW_QUAD = 4
MOTORS_PER_LEGEND_ROW_HEX = 3

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS_EDGE, "axes.labelcolor": SECONDARY, "xtick.color": SECONDARY,
    "ytick.color": SECONDARY, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})


def figure(rows, height_in, height_ratios=None):
    return plt.subplots(rows, 1, figsize=(FIGURE_WIDTH_IN, height_in),
                        gridspec_kw=dict(height_ratios=height_ratios) if height_ratios else None)


def motor_legend_columns(motor_count):
    return MOTORS_PER_LEGEND_ROW_QUAD if motor_count <= MOTORS_PER_LEGEND_ROW_QUAD else MOTORS_PER_LEGEND_ROW_HEX


def plot_motor_pwm(axis, time, pwm, line_width=1.2):
    for motor in range(pwm.shape[1]):
        axis.plot(time, pwm[:, motor], color=SERIES[motor], lw=line_width, label=f"Motor {motor + 1}")
    axis.legend(frameon=False, ncol=motor_legend_columns(pwm.shape[1]), loc="lower center")


def save(fig, path, vertical_padding=2):
    fig.tight_layout(h_pad=vertical_padding)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
