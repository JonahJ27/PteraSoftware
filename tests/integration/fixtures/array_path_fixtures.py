"""This module contains functions to run solvers on both the object path and the array
path, for comparing the two in the integration tests."""

import unittest
from collections.abc import Callable

import numpy as np

import pterasoftware as ps

# The largest relative difference allowed between the two paths' loads by default. The
# array path's vectorized norms round about one unit in the last place differently from
# the object path's, which changes the loads by about 1e-15 relative.
DEFAULT_RELATIVE_TOLERANCE = 1.0e-10

# The tolerance for aeroelastic problems with the Kutta-Joukowski force method. Once a
# Wing deforms, a trailing edge Panel's back leg center sits on the newest wake row's
# filament, where the induced velocity is extremely sensitive to the point's exact
# position, so a one unit in the last place change grows within a few time steps. Since
# the array path meshes each time step from its recorded step parameters (step 6 of the
# arrays rewrite), its geometry rounds differently from the object path's, and the loads
# differ by up to about 1.4e-8 relative (in the moments, whose lateral components are
# near zero). The object path alone differs by up to about 1.6e-8 when its wing density
# is nudged by 1e-15 relative, so this is the model's sensitivity, not a difference
# between the paths.
AEROELASTIC_JOUKOWSKI_RELATIVE_TOLERANCE = 1.0e-7

# The tolerances for the other coupled problems, each about ten times the largest
# relative difference the object path alone shows when its bound ring vortex strengths
# are nudged by 1e-15 relative at time step 1 (measured on each test's problem, which is
# also about the difference between the paths, so it is the problem's sensitivity, not a
# difference between the paths). The flap driven problem's sensitivity is about 4e-8 (in
# its moments), and the gliding free flight problem's about 2e-9. The flapping free
# flight problems are chaotic: the flapping one's sensitivity is about 3e-4 and the
# aeroelastic one's about 9e-5, so their tolerances are loose, and the gliding problem
# and the coupled problems' unit tests check the free flight array path more tightly.
FLAP_DRIVEN_RELATIVE_TOLERANCE = 1.0e-6
FREE_FLIGHT_GLIDER_RELATIVE_TOLERANCE = 1.0e-7
FREE_FLIGHT_FLAPPING_RELATIVE_TOLERANCE = 3.0e-3
FREE_FLIGHT_AEROELASTIC_RELATIVE_TOLERANCE = 1.0e-3

# The free flight flap driven problem is the most chaotic: its sensitivity is about 1e-3
# (in its moments). Its paths agree to round off (at most about 3e-16) through its last
# prescribed time step, and differ only once the body flies freely, so its prescribed
# time steps are checked at FREE_FLIGHT_FLAP_DRIVEN_HELD_RELATIVE_TOLERANCE and its
# whole run at FREE_FLIGHT_FLAP_DRIVEN_RELATIVE_TOLERANCE.
FREE_FLIGHT_FLAP_DRIVEN_HELD_RELATIVE_TOLERANCE = 1.0e-12
FREE_FLIGHT_FLAP_DRIVEN_RELATIVE_TOLERANCE = 1.0e-2

# The tolerance for a formation whose second Airplane moves and pitches. Its pitching
# Wing's geometry rounds a little differently on each path, and the loads differ by up
# to about 1.2e-10 relative, just above DEFAULT_RELATIVE_TOLERANCE (measured in M2's
# survey).
MOVING_FORMATION_RELATIVE_TOLERANCE = 1.0e-9


def make_solvers_run_on_both_paths(
    make_solver: Callable[
        [], ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
    ],
    prescribed_wake: bool = True,
    force_method: str = "joukowski",
) -> tuple[
    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
]:
    """This method makes a fixture that is two solvers of the same problem, one run on
    the object path and one on the array path.

    A free wake can amplify the paths' round off differences exponentially (a flapping
    Wing's rolled up wake is chaotic), so only problems whose free wake stays at round
    off over the whole run are compared with one.

    :param make_solver: A function that makes a new solver of the problem.
    :param prescribed_wake: Whether both runs use a prescribed wake. The default is
        True.
    :param force_method: The force method both runs use. The default is "joukowski".
    :return: The solver run with use_arrays=False, then the one run with
        use_arrays=True.
    """
    object_solver = make_solver()
    object_solver.run(
        prescribed_wake=prescribed_wake,
        show_progress=False,
        force_method=force_method,
        use_arrays=False,
    )
    array_solver = make_solver()
    array_solver.run(
        prescribed_wake=prescribed_wake,
        show_progress=False,
        force_method=force_method,
        use_arrays=True,
    )
    return object_solver, array_solver


def assert_loads_match(
    test_case: unittest.TestCase,
    object_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    array_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> None:
    """Asserts that two runs of the same problem, one on each path, have the same
    Airplane loads and load coefficients at every time step, and the same Panel loads
    and bound ring vortex strengths at the last time step, to round off.

    :param test_case: The TestCase making the assertions.
    :param object_solver: The solver run with use_arrays=False.
    :param array_solver: The solver run with use_arrays=True.
    :param relative_tolerance: The largest relative difference allowed. The default is
        DEFAULT_RELATIVE_TOLERANCE.
    :return: None
    """
    test_case.assertEqual(array_solver.num_steps, object_solver.num_steps)
    assert_airplane_loads_match(
        test_case,
        object_solver,
        array_solver,
        range(object_solver.num_steps),
        relative_tolerance,
    )
    object_strengths = object_solver._current_bound_vortex_strengths
    np.testing.assert_allclose(
        array_solver._current_bound_vortex_strengths,
        object_strengths,
        rtol=relative_tolerance,
        atol=relative_tolerance * float(np.max(np.abs(object_strengths))),
    )
    object_panel_forces = np.array(
        [panel.forces_GP1 for panel in object_solver.panels], dtype=float
    )
    np.testing.assert_allclose(
        np.array([panel.forces_GP1 for panel in array_solver.panels], dtype=float),
        object_panel_forces,
        rtol=relative_tolerance,
        atol=relative_tolerance * float(np.max(np.abs(object_panel_forces))),
    )


def assert_airplane_loads_match(
    test_case: unittest.TestCase,
    object_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    array_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    steps: range,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> None:
    """Asserts that two runs of the same problem, one on each path, have the same
    Airplane loads and load coefficients at each given time step.

    A chaotic problem's paths agree to round off only until its round off differences
    grow, so its early time steps can be checked with this more tightly than the whole
    run.

    :param test_case: The TestCase making the assertions.
    :param object_solver: The solver run with use_arrays=False.
    :param array_solver: The solver run with use_arrays=True.
    :param steps: The time steps to compare.
    :param relative_tolerance: The largest relative difference allowed. The default is
        DEFAULT_RELATIVE_TOLERANCE.
    :return: None
    """
    names = (
        "forces_W",
        "moments_W_CgP1",
        "forceCoefficients_W",
        "momentCoefficients_W_CgP1",
    )
    for step in steps:
        object_problem = object_solver._get_steady_problem_at(step)
        array_problem = array_solver._get_steady_problem_at(step)
        for object_airplane, array_airplane in zip(
            object_problem.airplanes, array_problem.airplanes
        ):
            for name in names:
                object_value = getattr(object_airplane, name)
                array_value = getattr(array_airplane, name)
                with test_case.subTest(step=step, name=name):
                    if object_value is None:
                        test_case.assertIsNone(array_value)
                        continue
                    np.testing.assert_allclose(
                        array_value,
                        object_value,
                        rtol=relative_tolerance,
                        atol=relative_tolerance * float(np.max(np.abs(object_value))),
                    )


# The public names whose values legitimately differ between the paths: which path ran.
_PATH_DEPENDENT_NAMES = frozenset({"use_arrays"})


def _public_names(obj: object) -> list[str]:
    """Returns the names of an object's public attributes and properties (not methods).

    :param obj: The object.
    :return: The names, sorted.
    """
    names = []
    for name in dir(type(obj)):
        if name.startswith("_") or name in _PATH_DEPENDENT_NAMES:
            continue
        class_attribute = getattr(type(obj), name, None)
        if callable(class_attribute) and not isinstance(class_attribute, property):
            continue
        names.append(name)
    return names


def _read(obj: object, name: str) -> object:
    """Returns an attribute's value, or the exception reading it raises.

    :param obj: The object.
    :param name: The attribute's name.
    :return: The value, or the exception.
    """
    try:
        return getattr(obj, name)
    except Exception as exception:  # noqa: BLE001
        return exception


def _is_number(value: object) -> bool:
    """Returns whether a value is a real number (not a bool).

    :param value: The value.
    :return: Whether it is a number.
    """
    return isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    )


def _magnitude(value: object) -> float:
    """Returns the largest magnitude of a number, a numeric ndarray, or a list of them,
    or 0.0 for anything else.

    :param value: The value.
    :return: The largest magnitude.
    """
    if _is_number(value):
        return abs(float(value))  # type: ignore[arg-type]
    if isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.size:
        return float(np.max(np.abs(value)))
    if isinstance(value, (list, tuple)):
        return max((_magnitude(item) for item in value), default=0.0)
    return 0.0


def _attribute_scales(root: object) -> dict[tuple[str, str], float]:
    """Returns, for each class and public attribute name, the largest magnitude that
    attribute has on any object of that class reachable from a root object.

    :param root: The object to start from.
    :return: The largest magnitudes, keyed by (class name, attribute name).
    """
    scales: dict[tuple[str, str], float] = {}
    visited: set[int] = set()
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        for name in _public_names(obj):
            value = _read(obj, name)
            key = (type(obj).__name__, name)
            scales[key] = max(scales.get(key, 0.0), _magnitude(value))
            children: list[object] = [value]
            if isinstance(value, (list, tuple)):
                children = list(value)
            elif isinstance(value, np.ndarray) and value.dtype == object:
                children = list(value.ravel())
            for child in children:
                if type(child).__module__.startswith("pterasoftware"):
                    stack.append(child)
    return scales


def assert_public_results_match(
    test_case: unittest.TestCase,
    object_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    array_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> int:
    """Asserts that everything a user can read from two runs of the same problem, one on
    each path, is the same, and returns how many values it compared.

    Starting from each solver, it follows every public attribute and property (names
    without a leading underscore) into every Ptera Software object they reach: the
    problem, its movement, every time step's SteadyProblem, OperatingPoint, Airplanes,
    Wings, WingCrossSections, and Panels, and their results. Objects that aren't Ptera
    Software's (such as MuJoCo's) and methods aren't followed. Only use_arrays may
    differ.

    Numbers and numeric ndarrays are compared to the given relative tolerance, relative
    to the largest of their own magnitude, their object's largest number, and the
    largest magnitude that attribute has anywhere in the object path's run (for example,
    every Panel's forces_GP1 at every time step). So a value that is round off, such as
    a symmetric problem's crosswind force coefficient, or the loads on a Panel that
    carries almost none, is compared on the scale of its kind. Everything else is
    compared exactly.

    :param test_case: The TestCase making the assertions.
    :param object_solver: The solver run with use_arrays=False.
    :param array_solver: The solver run with use_arrays=True.
    :param relative_tolerance: The largest relative difference allowed. The default is
        DEFAULT_RELATIVE_TOLERANCE.
    :return: The number of values compared.
    """
    scales = _attribute_scales(object_solver)
    visited: set[tuple[int, int]] = set()
    counter = {"values": 0}

    def compare(
        object_value: object, array_value: object, path: str, scale: float
    ) -> None:
        if object_value is None or array_value is None:
            test_case.assertIs(array_value is None, object_value is None, msg=path)
            return
        if isinstance(object_value, np.ndarray):
            test_case.assertIsInstance(array_value, np.ndarray, msg=path)
            assert isinstance(array_value, np.ndarray)
            test_case.assertEqual(array_value.shape, object_value.shape, msg=path)
            if object_value.dtype == object:
                for index, (object_item, array_item) in enumerate(
                    zip(object_value.ravel(), array_value.ravel())
                ):
                    compare(object_item, array_item, f"{path}[{index}]", scale)
                return
            counter["values"] += 1
            if object_value.dtype.kind in "biu" or object_value.size == 0:
                np.testing.assert_array_equal(array_value, object_value, err_msg=path)
                return
            np.testing.assert_allclose(
                array_value,
                object_value,
                rtol=relative_tolerance,
                atol=relative_tolerance * max(_magnitude(object_value), scale),
                err_msg=path,
            )
            return
        if isinstance(object_value, (bool, str, np.bool_)):
            counter["values"] += 1
            test_case.assertEqual(array_value, object_value, msg=path)
            return
        if _is_number(object_value):
            counter["values"] += 1
            test_case.assertTrue(
                np.isclose(
                    float(array_value),  # type: ignore[arg-type]
                    float(object_value),  # type: ignore[arg-type]
                    rtol=relative_tolerance,
                    atol=relative_tolerance * max(_magnitude(object_value), scale),
                ),
                msg=f"{path}: {array_value} != {object_value}",
            )
            return
        if isinstance(object_value, (list, tuple)):
            test_case.assertIsInstance(array_value, (list, tuple), msg=path)
            assert isinstance(array_value, (list, tuple))
            test_case.assertEqual(len(array_value), len(object_value), msg=path)
            for index, (object_item, array_item) in enumerate(
                zip(object_value, array_value)
            ):
                compare(object_item, array_item, f"{path}[{index}]", scale)
            return
        if isinstance(object_value, dict):
            test_case.assertIsInstance(array_value, dict, msg=path)
            assert isinstance(array_value, dict)
            test_case.assertEqual(set(array_value), set(object_value), msg=path)
            for key in object_value:
                compare(object_value[key], array_value[key], f"{path}[{key!r}]", scale)
            return
        if not type(object_value).__module__.startswith("pterasoftware"):
            return
        test_case.assertIs(type(array_value), type(object_value), msg=path)
        key = (id(object_value), id(array_value))
        if key in visited:
            return
        visited.add(key)
        outcomes = {
            name: (_read(object_value, name), _read(array_value, name))
            for name in _public_names(object_value)
        }
        object_scale = max(
            (
                _magnitude(object_outcome)
                for object_outcome, _ in outcomes.values()
                if _is_number(object_outcome)
            ),
            default=0.0,
        )
        class_name = type(object_value).__name__
        for name, (object_outcome, array_outcome) in outcomes.items():
            if isinstance(object_outcome, BaseException):
                test_case.assertIs(
                    type(array_outcome), type(object_outcome), msg=f"{path}.{name}"
                )
                continue
            compare(
                object_outcome,
                array_outcome,
                f"{path}.{name}",
                max(object_scale, scales.get((class_name, name), 0.0)),
            )

    compare(object_solver, array_solver, "solver", 0.0)
    return counter["values"]
