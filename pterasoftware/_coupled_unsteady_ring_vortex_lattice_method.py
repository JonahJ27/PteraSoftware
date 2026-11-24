"""Contains the CoupledUnsteadyRingVortexLatticeMethodSolver class."""

from __future__ import annotations

from typing import cast

import numpy as np

from . import _loads_arrays, _logging, _mesh_arrays, operating_point, problems
from .unsteady_ring_vortex_lattice_method import UnsteadyRingVortexLatticeMethodSolver

_logger = _logging.get_logger("_coupled_unsteady_ring_vortex_lattice_method")


class CoupledUnsteadyRingVortexLatticeMethodSolver(
    UnsteadyRingVortexLatticeMethodSolver
):
    """A subclass of UnsteadyRingVortexLatticeMethodSolver that solves
    _CoupledUnsteadyProblems.

    Geometry in a _CoupledUnsteadyProblem is determined step by step from the solver's
    results at the previous step, so bound vortices cannot be initialized upfront. This
    class inherits the parent's run() and initialize_step_geometry() unchanged and
    overrides three hooks: _initialize_step_vortices (per step bound vortex init),
    _update_next_step_hook (calls _CoupledUnsteadyProblem.initialize_next_problem
    between steps), and _get_steady_problem_at (dynamic dispatch through the problem's
    get_steady_problem accessor).
    """

    __slots__ = ()

    # _CoupledUnsteadyProblem is deliberately given a private class name inside the
    # public problems module so it stays out of the user-facing API. Referencing it from
    # this in-package solver is the intended design, so silence PyCharm's cross-module
    # protected-member inspection on the signature and the isinstance guard below.

    # noinspection PyProtectedMember
    def __init__(self, unsteady_problem: problems._CoupledUnsteadyProblem) -> None:
        """The initialization method.

        :param unsteady_problem: The _CoupledUnsteadyProblem to be solved.
        :return: None
        """
        if not isinstance(unsteady_problem, problems._CoupledUnsteadyProblem):
            raise TypeError("unsteady_problem must be a _CoupledUnsteadyProblem.")
        super().__init__(unsteady_problem)

    # As in __init__, the reference to the deliberately private _CoupledUnsteadyProblem
    # is by design; silence the cross-module protected-member inspection.

    # noinspection PyProtectedMember
    @property
    def _coupled_unsteady_problem(self) -> problems._CoupledUnsteadyProblem:
        """Type narrowed view of the inherited unsteady_problem attribute.

        The parent stores unsteady_problem as a CoreUnsteadyProblem (widened to let
        subclasses pass their own variants). __init__ validates that this subclass
        always receives a _CoupledUnsteadyProblem, so the cast here is safe.

        :return: The unsteady_problem narrowed to _CoupledUnsteadyProblem.
        """
        return cast(problems._CoupledUnsteadyProblem, self.unsteady_problem)

    def _initialize_step_vortices(self, step: int) -> None:
        _logger.debug(
            _logging.indent() + f"Initializing step {step}'s bound ring vortices"
        )
        self._initialize_panel_vortices_at(step)

    def _update_next_step_hook(self, step: int) -> None:
        self._coupled_unsteady_problem.initialize_next_problem(self, step)
        if step < self.num_steps - 1:
            self._initialize_panel_vortices_at(step + 1)

    def _get_steady_problem_at(self, step: int) -> problems.SteadyProblem:
        return self._coupled_unsteady_problem.get_steady_problem(step)

    def _operating_point_at(self, step: int) -> operating_point.OperatingPoint:
        """Gets the OperatingPoint to use for a given time step's geometry and wake.

        It replaces reading it from the time step's SteadyProblem, as the parent does.

        It exists because an array path run's time steps aren't built until they are
        read, so the OperatingPoint is read from the problem without building the time
        step. It is the same OperatingPoint.

        :param step: The time step.
        :return: The OperatingPoint at the time step.
        """
        return self._coupled_unsteady_problem.get_operating_point(step)

    def _store_step_results(
        self,
        step: int,
        loads: _loads_arrays.StepLoads | None,
        wake_corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        num_wake_rows: int,
        wing_shapes: tuple[tuple[int, int], ...],
    ) -> None:
        """Hands the array path's results at one time step to the problem, when the run
        ends.

        It replaces the parent's writes onto the time step's objects.

        It exists because a coupled problem builds an array path run's time steps only
        when they are read, so it keeps their results until then.

        :param step: The time step.
        :param loads: The time step's official StepLoads, or None if it finds no loads.
        :param wake_corners: The time step's wake ring vortex corners, or None at the
            first time step.
        :param num_wake_rows: The number of chordwise wake rows each Wing has at the
            time step.
        :param wing_shapes: Each Wing's (num_chordwise_panels, num_spanwise_panels), in
            global Wing order.
        :return: None
        """
        # noinspection PyProtectedMember
        self._coupled_unsteady_problem._store_array_results(
            step, loads, wake_corners, num_wake_rows, wing_shapes
        )

    def _wing_meshes_at(
        self, step: int
    ) -> tuple[list[_mesh_arrays.WingMesh], list[np.ndarray]]:
        """Returns the array path's meshes of every Wing at a given time step, and each
        Wing's Airplane's Cg_GP1_CgP1, both in global Wing order.

        It replaces reading each built Wing's mesh with
        _mesh_arrays.wing_mesh_from_wing.

        It exists because a coupled problem's geometry depends on the run (its
        structural deformation and flap angles), so the meshes can't come from the
        movement as the plain solver's do. They come from the step parameters the
        problem records as it builds each Airplane.

        A coupled problem builds each time step's Airplane during the run, with its
        structural deformation and flap angles, from step parameters it records. This
        meshes those same parameters, so the array path never reads the Wings. A free
        flight coupling trial changes only the OperatingPoint, not the geometry, so a
        trial at the next time step meshes the next time step's parameters.

        :param step: The time step.
        :return: The WingMeshes, and for each Wing, a (3,) ndarray of floats
            representing its Airplane's CG's position (in the first Airplane's geometry
            axes, relative to the first Airplane's CG). Its units are in meters.
        """
        # Changed for the arrays rewrite (step 6): the meshes used to be read from the
        # Wings the problem built, with _mesh_arrays.wing_mesh_from_wing, which is no
        # longer called here. The old code was:
        # octowrap: off
        # wing_meshes: list[_mesh_arrays.WingMesh] = []
        # wingsCg_GP1_CgP1: list[np.ndarray] = []
        # for airplane in self._get_steady_problem_at(step).airplanes:
        #     for wing in airplane.wings:
        #         wing_meshes.append(_mesh_arrays.wing_mesh_from_wing(wing))
        #         wingsCg_GP1_CgP1.append(airplane.Cg_GP1_CgP1)
        # return wing_meshes, wingsCg_GP1_CgP1
        # octowrap: on
        return self._wing_meshes_from_parameters(
            [self._coupled_unsteady_problem.get_airplane_step_parameters(step)]
        )
