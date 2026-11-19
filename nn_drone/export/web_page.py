"""Build the self-contained web pages: web/index.html (3D flight range) and web/network.html (network lab),
each with its JavaScript core and both shipped networks inside."""
import json

from nn_drone.export.model_package import flight_network_arrays
from nn_drone.paths import DEFAULT_WEIGHTS, WEB_DIR
from nn_drone.simulation.config import SimulatorConfig
from nn_drone.simulation.frames import layout_named

DECIMALS = 6
BYTES_PER_KB = 1024
CORE_PLACEHOLDER = "/*__CORE__*/"
NETWORK_PLACEHOLDERS = {"quad_x": "/*__WQ__*/", "hex_x": "/*__WH__*/"}
TIMING_KEY = "timing"
LAB_CORE_PLACEHOLDER = "/*__LAB_CORE__*/"
LAB_NETWORKS_PLACEHOLDER = "/*__NETWORKS__*/"


def browser_network(weights_path):
    """The flight network as the JSON object web/core.js expects: W0, b0, W1, b1, W2, b2 and its timing."""
    info, arrays = flight_network_arrays(weights_path)
    network = {key.removeprefix("pi_"): arrays[key].tolist() for key in arrays}
    network[TIMING_KEY] = browser_timing(info.profile.control_timing)
    return network


def browser_timing(timing):
    return dict(period=timing.period_s, scalesWithSize=timing.scales_with_size, substeps=timing.physics_substeps)


def _rounded(values):
    if isinstance(values, list):
        return [_rounded(value) for value in values]
    return round(float(values), DECIMALS)


def build_web_page():
    page = (WEB_DIR / "template.html").read_text(encoding="utf-8")
    page = page.replace(CORE_PLACEHOLDER, (WEB_DIR / "core.js").read_text(encoding="utf-8"))
    page = page.replace(LAB_CORE_PLACEHOLDER, (WEB_DIR / "network_core.js").read_text(encoding="utf-8"))
    for frame, placeholder in NETWORK_PLACEHOLDERS.items():
        network = {name: values if name == TIMING_KEY else _rounded(values)
                   for name, values in browser_network(DEFAULT_WEIGHTS[frame]).items()}
        page = page.replace(placeholder, json.dumps(network, separators=(",", ":")))
    output = WEB_DIR / "index.html"
    output.write_text(page, encoding="utf-8")
    return output, len(page) // BYTES_PER_KB


def lab_network(frame):
    """The browser network plus what the lab needs to explain its outputs: motor layout and thrust."""
    layout, config = layout_named(frame), SimulatorConfig()
    network = {name: values if name == TIMING_KEY else _rounded(values)
               for name, values in browser_network(DEFAULT_WEIGHTS[frame]).items()}
    network.update(layout=dict(angles=list(layout.angles_from_nose_deg), spin=list(layout.spin_directions)),
                   thrustToWeight=config.thrust_to_weight, nominalBatteryV=config.nominal_battery_voltage)
    return network


def build_network_page():
    page = (WEB_DIR / "network_template.html").read_text(encoding="utf-8")
    page = page.replace(LAB_CORE_PLACEHOLDER, (WEB_DIR / "network_core.js").read_text(encoding="utf-8"))
    networks = {frame: lab_network(frame) for frame in NETWORK_PLACEHOLDERS}
    page = page.replace(LAB_NETWORKS_PLACEHOLDER, json.dumps(networks, separators=(",", ":")))
    output = WEB_DIR / "network.html"
    output.write_text(page, encoding="utf-8")
    return output, len(page) // BYTES_PER_KB
