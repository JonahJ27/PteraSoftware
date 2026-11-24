"""This module contains functions for testing the step parameters coupled unsteady
problems record for the solvers' array path."""

import unittest
from typing import Any

import numpy as np

import pterasoftware as ps


def assert_recorded_step_parameters_match_the_airplanes(
    test_case: unittest.TestCase,
    problem: Any,
    airplane_movement: Any,
    steps: range,
) -> None:
    """Asserts that, at each given time step, the Airplane built from the step
    parameters the problem recorded is exactly the first Airplane the problem solved at
    that time step: the same CG and the same Panel corners for every Wing.

    :param test_case: The TestCase making the assertions.
    :param problem: The solved coupled unsteady problem.
    :param airplane_movement: The first Airplane's movement, whose
        _airplane_from_parameters rebuilds the Airplane.
    :param steps: The time steps to check.
    :return: None
    """
    for step in steps:
        with test_case.subTest(step=step):
            rebuilt: ps.geometry.airplane.Airplane = (
                airplane_movement._airplane_from_parameters(
                    problem.get_airplane_step_parameters(step)
                )
            )
            solved: ps.geometry.airplane.Airplane = problem.steady_problems[
                step
            ].airplanes[0]
            test_case.assertTrue(
                np.array_equal(rebuilt.Cg_GP1_CgP1, solved.Cg_GP1_CgP1)
            )
            test_case.assertEqual(len(rebuilt.wings), len(solved.wings))
            for rebuilt_wing, solved_wing in zip(rebuilt.wings, solved.wings):
                rebuilt_panels = rebuilt_wing.panels
                solved_panels = solved_wing.panels
                assert rebuilt_panels is not None
                assert solved_panels is not None
                test_case.assertEqual(rebuilt_panels.shape, solved_panels.shape)
                for rebuilt_panel, solved_panel in zip(
                    rebuilt_panels.ravel(), solved_panels.ravel()
                ):
                    for corner in ("Frpp_G_Cg", "Flpp_G_Cg", "Blpp_G_Cg", "Brpp_G_Cg"):
                        test_case.assertTrue(
                            np.array_equal(
                                getattr(rebuilt_panel, corner),
                                getattr(solved_panel, corner),
                            )
                        )
