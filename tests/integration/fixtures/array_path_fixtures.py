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
# position, so a one unit in the last place change grows to about 6e-10 relative within
# a time step. The object path alone does the same when its wing density is nudged by
# 1e-15 relative, so this is the model's sensitivity, not a difference between the
# paths.
AEROELASTIC_JOUKOWSKI_RELATIVE_TOLERANCE = 1.0e-8


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
    names = (
        "forces_W",
        "moments_W_CgP1",
        "forceCoefficients_W",
        "momentCoefficients_W_CgP1",
    )
    test_case.assertEqual(array_solver.num_steps, object_solver.num_steps)
    for step in range(object_solver.num_steps):
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
