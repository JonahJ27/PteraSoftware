"""This module contains functions to run solvers on both the object path and the array
path, for comparing the two in the integration tests."""

import unittest
from collections.abc import Callable

import numpy as np

import pterasoftware as ps

# The largest relative difference allowed between the two paths' loads. The array path's
# vectorized norms round about one unit in the last place differently from the object
# path's, which changes the loads by about 1e-15 relative.
_RELATIVE_TOLERANCE = 1.0e-10


def make_solvers_run_on_both_paths(
    make_solver: Callable[
        [], ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
    ],
    prescribed_wake: bool = True,
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
    :return: The solver run with use_arrays=False, then the one run with
        use_arrays=True.
    """
    object_solver = make_solver()
    object_solver.run(
        prescribed_wake=prescribed_wake, show_progress=False, use_arrays=False
    )
    array_solver = make_solver()
    array_solver.run(
        prescribed_wake=prescribed_wake, show_progress=False, use_arrays=True
    )
    return object_solver, array_solver


def assert_loads_match(
    test_case: unittest.TestCase,
    object_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
    array_solver: ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver,
) -> None:
    """Asserts that two runs of the same problem, one on each path, have the same
    Airplane loads and load coefficients at every time step, and the same Panel loads
    and bound ring vortex strengths at the last time step, to round off.

    :param test_case: The TestCase making the assertions.
    :param object_solver: The solver run with use_arrays=False.
    :param array_solver: The solver run with use_arrays=True.
    :return: None
    """
    names = (
        "forces_W",
        "moments_W_CgP1",
        "forceCoefficients_W",
        "momentCoefficients_W_CgP1",
    )
    for step, (object_problem, array_problem) in enumerate(
        zip(object_solver.steady_problems, array_solver.steady_problems)
    ):
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
                        rtol=_RELATIVE_TOLERANCE,
                        atol=_RELATIVE_TOLERANCE * float(np.max(np.abs(object_value))),
                    )

    object_strengths = object_solver._current_bound_vortex_strengths
    np.testing.assert_allclose(
        array_solver._current_bound_vortex_strengths,
        object_strengths,
        rtol=_RELATIVE_TOLERANCE,
        atol=_RELATIVE_TOLERANCE * float(np.max(np.abs(object_strengths))),
    )
    object_panel_forces = np.array(
        [panel.forces_GP1 for panel in object_solver.panels], dtype=float
    )
    np.testing.assert_allclose(
        np.array([panel.forces_GP1 for panel in array_solver.panels], dtype=float),
        object_panel_forces,
        rtol=_RELATIVE_TOLERANCE,
        atol=_RELATIVE_TOLERANCE * float(np.max(np.abs(object_panel_forces))),
    )
