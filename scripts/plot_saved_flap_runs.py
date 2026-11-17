"""Plots saved runs of the free flight flap driven examples, one run alone or several
overlaid for comparison.

The free flight flap examples save each run (with SAVE_RESULTS) in its own folder,
examples/saved_results/<example name>/<date and time>_wake_<rows>/, holding
metadata.json (the run's settings) and results.npz (its time histories). This script
reads those two files, so it doesn't need the large solver.psz.

Usage:

python scripts/plot_saved_flap_runs.py     Plots the newest run of each example. python
scripts/plot_saved_flap_runs.py RUN_FOLDER [RUN_FOLDER ...]     Overlays the given runs.
python scripts/plot_saved_flap_runs.py --latest 3     Overlays the newest 3 runs (add
--example NAME to pick one example's runs). python scripts/plot_saved_flap_runs.py ...
--save     Also saves the figures as PNG files, in examples/saved_results/plots/<date
and     time>/.

Each run's legend label names its example and time, and the settings in which it differs
from the other runs, so that, for example, runs with and without a wake limit are told
apart. The script also prints each run's settings and summary numbers and, for two or
more runs, the largest difference of each history from the first run's.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_RESULTS_DIR = PROJECT_ROOT / "examples" / "saved_results"

# These are the metadata settings that a run's label can show when they differ between
# the runs being plotted, with the short name used in the label.
_LABEL_SETTINGS = {
    "max_wake_rows": "wake",
    "flap_frequency_hz": "f",
    "air_speed_m_per_s": "V",
    "angle_of_attack_deg": "AoA",
    "steps_per_flap": "steps/flap",
    "free_num_flaps": "free flaps",
    "mass_kg": "m",
    "spring_constant_rad": "k",
    "damping_constant_rad": "c",
    "wing_density_kg_per_m2": "density",
    "use_analytical_spring_solver": "analytical",
}


class SavedRun:
    """One saved run: its folder, its metadata, and its results."""

    def __init__(self, folder: Path) -> None:
        """Loads a saved run.

        :param folder: The run's folder, which holds metadata.json and results.npz.
        :return: None
        """
        self.folder = folder
        with open(folder / "metadata.json", encoding="utf-8") as file:
            self.metadata: dict[str, Any] = json.load(file)
        with np.load(folder / "results.npz") as results:
            self.results = {name: results[name] for name in results.files}
        self.label = folder.name

    @property
    def free_flight_start_s(self) -> float:
        """The time at which free flight starts (s)."""
        return float(self.results["prescribed_num_steps"]) * float(
            self.metadata["delta_time_s"]
        )


def _find_runs(arguments: argparse.Namespace) -> list[Path]:
    """Finds the run folders to plot.

    :param arguments: The parsed command line arguments.
    :return: The run folders, oldest first.
    """
    if arguments.runs:
        folders = [Path(run).resolve() for run in arguments.runs]
        for folder in folders:
            if not (folder / "results.npz").exists():
                sys.exit(f"{folder} has no results.npz. Is it a saved run's folder?")
        return folders

    if not SAVED_RESULTS_DIR.exists():
        sys.exit(f"No saved runs yet: {SAVED_RESULTS_DIR} doesn't exist.")
    example_folders = sorted(
        folder
        for folder in SAVED_RESULTS_DIR.iterdir()
        if folder.is_dir() and folder.name != "plots"
    )
    if arguments.example is not None:
        example_folders = [
            folder for folder in example_folders if folder.name == arguments.example
        ]

    def runs_in(example_folder: Path) -> list[Path]:
        return sorted(
            run for run in example_folder.iterdir() if (run / "results.npz").exists()
        )

    if arguments.latest is not None:
        all_runs = sorted(
            (run for folder in example_folders for run in runs_in(folder)),
            key=lambda run: run.name,
        )
        return all_runs[-arguments.latest :]
    return [runs_in(folder)[-1] for folder in example_folders if runs_in(folder)]


def _label_runs(runs: list[SavedRun]) -> None:
    """Gives each run a legend label: its example, its time, and the settings that
    differ between the runs.

    :param runs: The runs.
    :return: None
    """
    examples = {run.metadata["example"] for run in runs}
    differing = [
        key
        for key in _LABEL_SETTINGS
        if len({json.dumps(run.metadata.get(key)) for run in runs}) > 1
    ]
    for run in runs:
        parts = []
        if len(examples) > 1:
            # Name the vehicle from the parameter files the run loaded.
            parameter_files = run.metadata.get("parameter_files", {})
            parts.append(
                "FAAV"
                if any(name.startswith("faav") for name in parameter_files)
                else "pterosaur"
            )
        parts.append(run.metadata["saved_at"].replace("T", " ")[5:16])
        for key in differing:
            value = run.metadata.get(key)
            parts.append(f"{_LABEL_SETTINGS[key]} {'full' if value is None else value}")
        run.label = ", ".join(parts)


def _print_summary(runs: list[SavedRun]) -> None:
    """Prints each run's settings and summary numbers, and the differences from the
    first run.

    :param runs: The runs.
    :return: None
    """
    for run in runs:
        meta, results = run.metadata, run.results
        free = np.arange(results["times_s"].size) >= int(
            results["prescribed_num_steps"]
        )
        torques = np.abs(results["motor_torques_N_m"])
        limits = results["motor_max_torques_N_m"]
        at_limit = np.mean(torques >= limits * (1.0 - 1e-6)) * 100.0
        print(f"\n{run.label}  ({run.folder})")
        print(
            f"  {meta['example']}: {meta['flap_frequency_hz']} Hz, "
            f"{meta['air_speed_m_per_s']} m/s, {meta['angle_of_attack_deg']} deg, "
            f"{meta['steps_per_flap']} steps/flap, {meta['prescribed_num_flaps']} + "
            f"{meta['free_num_flaps']} flaps, wake "
            f"{'full' if meta['max_wake_rows'] is None else meta['max_wake_rows']}, "
            f"mass {meta['mass_kg']} kg, k {meta['spring_constant_rad']}, "
            f"c {meta['damping_constant_rad']}, run time "
            f"{meta['simulation_run_time_s'] / 60.0:.1f} min"
        )
        lift = -results["forces_W_N"][:, 2]
        weight = meta["mass_kg"] * 9.80665
        print(
            f"  free flight: mean lift {np.nanmean(lift[free]):.3g} N "
            f"(weight {weight:.3g} N), peak motor torque {torques.max():.3g} N*m, "
            f"motor at its limit {at_limit:.0f}% of the time, final speed "
            f"{results['speeds_m_per_s'][-1]:.3g} m/s, final pitch "
            f"{results['angles_E_to_BP1_izyx_deg'][-1, 1]:.3g} deg, altitude change "
            f"{-(results['positions_CgP1_E_Eo_m'][-1, 2] - results['positions_CgP1_E_Eo_m'][0, 2]):.3g} m, "
            f"max tip twist "
            f"{np.rad2deg(np.max(np.abs(results['deformation_angles_y_rad'][:, :, -1]))):.3g} deg"
        )

    if len(runs) < 2:
        return
    reference = runs[0]
    print(
        f"\nLargest difference from the first run ({reference.label}), over the time "
        "they share, as a fraction of the first run's largest value:"
    )
    histories = {
        "flap angle": lambda r: r["flap_angles_rad"][0],
        "motor torque": lambda r: r["motor_torques_N_m"][0],
        "fluid torque": lambda r: r["fluid_torques_N_m"][0],
        "tip twist": lambda r: r["deformation_angles_y_rad"][0, :, -1],
        "lift": lambda r: -r["forces_W_N"][:, 2],
        "speed": lambda r: r["speeds_m_per_s"],
        "pitch angle": lambda r: r["angles_E_to_BP1_izyx_deg"][:, 1],
        "altitude": lambda r: -r["positions_CgP1_E_Eo_m"][:, 2],
    }
    for run in runs[1:]:
        print(f"  {run.label}:")
        for name, history in histories.items():
            times = run.results["times_s"]
            shared = times <= min(times[-1], reference.results["times_s"][-1])
            expected = np.interp(
                times[shared], reference.results["times_s"], history(reference.results)
            )
            actual = history(run.results)[shared]
            scale = max(float(np.nanmax(np.abs(expected))), 1e-12)
            difference = float(np.nanmax(np.abs(actual - expected))) / scale
            print(f"    {name:13s} {difference:8.1e}")


def _plot(runs: list[SavedRun]) -> list[plt.Figure]:
    """Plots the runs.

    :param runs: The runs.
    :return: The figures.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figures = []

    def mark_free_flight(axes: Any) -> None:
        for axis in np.ravel(axes):
            for run, color in zip(runs, colors):
                axis.axvline(
                    run.free_flight_start_s, color=color, linestyle=":", alpha=0.5
                )
            axis.grid(True)

    # The flap drives: the flap angle, the motor's torque against its limit, the motor
    # speed, and the fluid torque on the drive's Wings, for each run's first drive.
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, num="Flap Drive")
    for run, color in zip(runs, colors):
        r, t = run.results, run.results["times_s"]
        axes[0, 0].plot(
            t, np.rad2deg(r["flap_angles_rad"][0]), color=color, label=run.label
        )
        axes[0, 1].plot(t, r["motor_torques_N_m"][0], color=color, label=run.label)
        axes[0, 1].plot(
            t, r["motor_max_torques_N_m"][0], color=color, linestyle="--", linewidth=0.8
        )
        axes[0, 1].plot(
            t,
            -r["motor_max_torques_N_m"][0],
            color=color,
            linestyle="--",
            linewidth=0.8,
        )
        axes[1, 0].plot(t, r["motor_velocities_rad"][0], color=color, label=run.label)
        axes[1, 1].plot(t, r["fluid_torques_N_m"][0], color=color, label=run.label)
    axes[0, 0].set_ylabel("Flap angle [deg]")
    axes[0, 1].set_ylabel("Motor torque [N*m] (dashed: limit)")
    axes[1, 0].set_ylabel("Motor speed [rad/s]")
    axes[1, 1].set_ylabel("Fluid torque on the wings [N*m]")
    for axis in axes[1]:
        axis.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8)
    mark_free_flight(axes)
    figure.suptitle("Flap drive (first drive; dotted lines: start of free flight)")
    figure.tight_layout()
    figures.append(figure)

    # The vehicle: its path, speed, altitude, angle of attack and pitch angle, and body
    # rates.
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), num="Vehicle")
    for run, color in zip(runs, colors):
        r, t = run.results, run.results["times_s"]
        positions = r["positions_CgP1_E_Eo_m"]
        axes[0, 0].plot(positions[:, 0], -positions[:, 2], color=color, label=run.label)
        axes[0, 1].plot(t, r["speeds_m_per_s"], color=color)
        axes[1, 0].plot(t, -positions[:, 2], color=color)
        axes[1, 1].plot(t, r["alphas_deg"], color=color)
        axes[1, 1].plot(
            t, r["angles_E_to_BP1_izyx_deg"][:, 1], color=color, linestyle="--"
        )
        omegas = r["omegas_BP1__E_deg_per_s"]
        axes[2, 0].plot(t, omegas[:, 1], color=color)
        axes[2, 1].plot(t, omegas[:, 0], color=color)
        axes[2, 1].plot(t, omegas[:, 2], color=color, linestyle="--")
    axes[0, 0].set_xlabel("Downrange (Earth x) [m]")
    axes[0, 0].set_ylabel("Altitude [m]")
    axes[0, 0].set_title("Flight path")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)
    axes[0, 1].set_ylabel("Speed [m/s]")
    axes[1, 0].set_ylabel("Altitude [m]")
    axes[1, 1].set_ylabel("Angle of attack (solid), pitch (dashed) [deg]")
    axes[2, 0].set_ylabel("Pitch rate [deg/s]")
    axes[2, 1].set_ylabel("Roll (solid), yaw (dashed) rate [deg/s]")
    for axis in (axes[2, 0], axes[2, 1]):
        axis.set_xlabel("Time [s]")
    mark_free_flight([axes[0, 1], *np.ravel(axes[1:])])
    figure.suptitle("Vehicle (dotted lines: start of free flight)")
    figure.tight_layout()
    figures.append(figure)

    # The loads and the twist: lift, drag, pitching moment, and each main Wing's tip
    # twist.
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 8), sharex=True, num="Loads and Twist"
    )
    for run, color in zip(runs, colors):
        r, t = run.results, run.results["times_s"]
        axes[0, 0].plot(t, -r["forces_W_N"][:, 2], color=color, label=run.label)
        axes[0, 1].plot(t, -r["forces_W_N"][:, 0], color=color)
        axes[1, 0].plot(t, r["moments_W_CgP1_N_m"][:, 1], color=color)
        twist = np.rad2deg(r["deformation_angles_y_rad"][:, :, -1])
        axes[1, 1].plot(t, twist[0], color=color)
        if twist.shape[0] > 1:
            axes[1, 1].plot(t, twist[1], color=color, linestyle="--")
    weight_lines = {run.metadata["mass_kg"] * 9.80665 for run in runs}
    for weight in weight_lines:
        axes[0, 0].axhline(weight, color="k", linewidth=0.8)
    axes[0, 0].set_ylabel("Lift [N] (black: weight)")
    axes[0, 1].set_ylabel("Drag [N]")
    axes[1, 0].set_ylabel("Pitching moment [N*m]")
    axes[1, 1].set_ylabel("Tip twist [deg] (solid: wing 0, dashed: wing 1)")
    for axis in axes[1]:
        axis.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8)
    mark_free_flight(axes)
    figure.suptitle("Loads and twist (dotted lines: start of free flight)")
    figure.tight_layout()
    figures.append(figure)

    return figures


def main() -> int:
    """Finds, summarizes, and plots saved runs.

    :return: 0.
    """
    parser = argparse.ArgumentParser(
        description="Plot saved runs of the free flight flap driven examples."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run folders to plot. Default: each example's newest run.",
    )
    parser.add_argument(
        "--latest", type=int, default=None, help="Plot the newest N saved runs."
    )
    parser.add_argument(
        "--example",
        default=None,
        help="Only consider this example's runs (its folder name in saved_results).",
    )
    parser.add_argument(
        "--save", action="store_true", help="Also save the figures as PNG files."
    )
    arguments = parser.parse_args()

    runs = [SavedRun(folder) for folder in _find_runs(arguments)]
    if not runs:
        sys.exit("No saved runs found.")
    _label_runs(runs)
    _print_summary(runs)
    figures = _plot(runs)

    if arguments.save:
        plots_folder = (
            SAVED_RESULTS_DIR / "plots" / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        )
        plots_folder.mkdir(parents=True, exist_ok=True)
        for figure in figures:
            name = str(figure.get_label()).lower().replace(" ", "_")
            figure.savefig(plots_folder / f"{name}.png", dpi=110)
        print(f"\nSaved the figures to {plots_folder}.")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
