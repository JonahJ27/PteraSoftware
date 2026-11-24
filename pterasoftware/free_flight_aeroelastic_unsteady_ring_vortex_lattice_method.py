"""Contains the FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver class."""

from __future__ import annotations

import logging
from typing import NamedTuple, cast

import numpy as np

from . import _functions, geometry, operating_point, problems
from .aeroelastic_unsteady_ring_vortex_lattice_method import (
    AeroelasticUnsteadyRingVortexLatticeMethodSolver,
)

# The base solver's logger, so the wake grid singularity events this solver now logs in
# freeze_substep read exactly as the official wake build logged them before. It is
# imported rather than created here, because the logging setup sizes its name column
# from the modules that create a logger, and this module does not log under its own
# name.
from .unsteady_ring_vortex_lattice_method import _logger

# Body and geometry axes differ by a 180-degree rotation about y, so transforming a free
# vector (such as an angular velocity) from the first Airplane's body axes to its
# geometry axes negates the x and z components.
_BP1_TO_GP1_FLIP = np.array([-1.0, 1.0, -1.0], dtype=float)
_BP1_TO_GP1_FLIP.flags.writeable = False


class _FinalTrial(NamedTuple):
    """The final coupling trial's solution at the next time step, kept by
    keep_final_trial for that time step's official solve.

    **Contains the following fields:**

    step: The time step the trial solved, which is the next time step of the coupling
    that kept it.

    strengths: A (N,) ndarray of floats of the trial's bound ring vortex strengths. The
    units are in meters squared per second.

    grid_wing_wing_influences: A (N,N) ndarray of floats of the trial's Wing Wing
    influence coefficients (observed from the Earth frame).

    stack_freestream_wing_influences: A (N,) ndarray of floats of the trial's freestream
    Wing influence coefficients (observed from the Earth frame).

    stack_wake_wing_influences: A (N,) ndarray of floats of the trial's wake Wing
    influence coefficients (observed from the Earth frame).

    stackPanelForces_GP1: A (N,3) ndarray of floats of the trial's forces (in the first
    Airplane's geometry axes) on every Panel. The units are in Newtons.

    stackPanelMoments_GP1_CgP1: A (N,3) ndarray of floats of the trial's moments (in the
    first Airplane's geometry axes, relative to the first Airplane's CG) on every Panel.
    The units are in Newton-meters.
    """

    step: int
    strengths: np.ndarray
    grid_wing_wing_influences: np.ndarray
    stack_freestream_wing_influences: np.ndarray
    stack_wake_wing_influences: np.ndarray
    stackPanelForces_GP1: np.ndarray
    stackPanelMoments_GP1_CgP1: np.ndarray


class FreeFlightAeroelasticUnsteadyRingVortexLatticeMethodSolver(
    AeroelasticUnsteadyRingVortexLatticeMethodSolver
):
    """A class used to solve FreeFlightAeroelasticUnsteadyProblem with the unsteady ring
    vortex lattice method.

    In a FreeFlightAeroelasticUnsteadyProblem the body state at each time step comes
    from the coupled MuJoCo rigid-body dynamics (carried out in
    FreeFlightAeroelasticUnsteadyProblem's initialize_next_problem), so each step's
    OperatingPoint carries a body angular rate (omegas_BP1__E) that the standard solver
    assumes is zero. This solver contributes the apparent velocity from that body rate
    (omega cross r) at every collocation point and bound line vortex leg center.

    **Key additions over the unsteady ring vortex lattice method:**

    Permits non-zero body rates, and supplies the current and next OperatingPoints' body
    angular rates (in the first Airplane's geometry axes, in radians per second) to the
    velocity calculations, the latter used when convecting the wake to the next time
    step.

    **Strongly coupled solve interface:**

    freeze_substep, evaluate_trial_aero_loads, and restore_substep are the public
    methods the FreeFlightAeroelasticUnsteadyProblem's step solve drives to evaluate the
    aerodynamics at successive trial body states within a single time step. They share
    transient working state (the frozen wake induced velocities and the snapshot bound
    ring vortex strengths) that freeze_substep establishes and restore_substep clears.
    """

    __slots__ = (
        "_substep_next_step",
        "_substep_next_steady_problem",
        "_substep_next_operating_point",
        "_substepStackVIndGridWrvp_GP1__E",
        "_substep_gamma_n",
        "_substep_gamma_n_minus_1",
        "_substep_saved_state",
        "_substep_final_trial",
        "_stepStackPanelForces_GP1",
        "_stepStackPanelMoments_GP1_CgP1",
    )

    _models_body_rates = True

    def __init__(
        self,
        free_flight_unsteady_problem: problems.FreeFlightAeroelasticUnsteadyProblem,
    ) -> None:
        """The initialization method.

        :param free_flight_unsteady_problem: The FreeFlightAeroelasticUnsteadyProblem to
            be solved.
        :return: None
        """
        if not isinstance(
            free_flight_unsteady_problem, problems.FreeFlightAeroelasticUnsteadyProblem
        ):
            raise TypeError(
                "free_flight_unsteady_problem must be a FreeFlightAeroelasticUnsteadyProblem."
            )

        # This is transient working state for the strongly coupled sub-iteration,
        # established by freeze_substep and cleared by restore_substep. Each is None
        # between substeps. The next step and its transient SteadyProblem and trial
        # OperatingPoint redirect the inherited geometry and wake reads to a scratch
        # copy of the next Airplane and the current trial state, since the canonical
        # next-step SteadyProblem is not committed until the solve accepts. The frozen
        # induced velocities are the iterate-independent part of the wake transport. The
        # two strength snapshots are the current and previous steps' solved bound ring
        # vortex strengths. These are initialized before the inherited constructor runs,
        # because it calls _get_steady_problem_at, which this solver's override consults
        # for the substep state.
        self._substep_next_step: int | None = None
        self._substep_next_steady_problem: problems.SteadyProblem | None = None
        self._substep_next_operating_point: operating_point.OperatingPoint | None = None
        self._substepStackVIndGridWrvp_GP1__E: list[list[np.ndarray]] | None = None
        self._substep_gamma_n: np.ndarray | None = None
        self._substep_gamma_n_minus_1: np.ndarray | None = None
        # The saved solver state of the step being solved, keyed by attribute name.
        # freeze_substep saves it before the trials overwrite the solver's attributes,
        # and restore_substep puts it back afterwards.
        self._substep_saved_state: dict[str, object] | None = None
        # The final coupling trial's solution, kept by keep_final_trial for the next
        # time step's official solve when the problem's Approximations has
        # reuse_final_trial, or None. Its name starts with _substep so that
        # restore_substep, which puts back every other slot, leaves it alone: it must
        # survive restore_substep and the official wake build until the next time step's
        # official solve uses it.
        self._substep_final_trial: _FinalTrial | None = None
        # These are the per Panel forces (in the first Airplane's geometry axes) and
        # moments (in the first Airplane's geometry axes, relative to the first
        # Airplane's CG) of the most recent load calculation, recorded by
        # _process_panel_loads_hook so keep_final_trial can keep the final trial's. Each
        # is an (N,3) ndarray of floats, or None before the first load calculation.
        self._stepStackPanelForces_GP1: np.ndarray | None = None
        self._stepStackPanelMoments_GP1_CgP1: np.ndarray | None = None

        super().__init__(free_flight_unsteady_problem)

    @property
    def _free_flight_unsteady_problem(
        self,
    ) -> problems.FreeFlightAeroelasticUnsteadyProblem:
        """The solver's FreeFlightAeroelasticUnsteadyProblem, narrowed from the
        inherited unsteady_problem.

        The inherited unsteady_problem slot is typed as the base CoreUnsteadyProblem so
        the parent solver can hold any coupled problem. This solver's constructor only
        accepts a FreeFlightAeroelasticUnsteadyProblem, so the cast here is safe.

        :return: This solver's FreeFlightAeroelasticUnsteadyProblem.
        """
        return cast(
            problems.FreeFlightAeroelasticUnsteadyProblem, self.unsteady_problem
        )

    def _get_steady_problem_at(self, step: int) -> problems.SteadyProblem:
        """Gets the SteadyProblem at a given time step.

        During a strongly coupled sub-iteration, returns the transient next-step
        SteadyProblem, which is built over a scratch copy of the prescribed next-step
        Airplane so the canonical Airplane's set-once panel coordinates are reserved for
        the official SteadyProblem committed once the solve accepts. Otherwise defers to
        the inherited coupled accessor.

        :param step: The time step of the desired SteadyProblem.
        :return: The SteadyProblem at the given time step.
        """
        if (
            self._substep_next_steady_problem is not None
            and step == self._substep_next_step
        ):
            return self._substep_next_steady_problem
        return super()._get_steady_problem_at(step)

    def _operating_point_at(self, step: int) -> operating_point.OperatingPoint:
        """Gets the OperatingPoint to use for a given time step's geometry and wake.

        During a strongly coupled sub-iteration, returns the current trial
        OperatingPoint for the next step, which is not bound to any committed
        SteadyProblem (the transient next-step SteadyProblem carries a placeholder).
        Otherwise defers to the inherited accessor.

        :param step: The time step of the desired OperatingPoint.
        :return: The OperatingPoint to use at the given time step.
        """
        if (
            self._substep_next_operating_point is not None
            and step == self._substep_next_step
        ):
            return self._substep_next_operating_point
        return super()._operating_point_at(step)

    def _currentOmegasRad_GP1__E(self) -> np.ndarray:
        """Finds the current time step's body angular velocity (in the first Airplane's
        geometry axes, observed from the Earth frame).

        The current OperatingPoint stores the body angular rate in the first Airplane's
        body axes, in degrees per second. This method transforms it to the first
        Airplane's geometry axes and converts it to radians per second for the omega
        cross r calculation in _apply_body_rate.

        :return: A (3,) ndarray of floats representing the body angular velocity (in the
            first Airplane's geometry axes, observed from the Earth frame). Its units
            are in radians per second.
        """
        omegas_GP1__E = self.current_operating_point.omegas_BP1__E * _BP1_TO_GP1_FLIP
        return cast(np.ndarray, np.deg2rad(omegas_GP1__E))

    def _convectionOmegasRad_GP1__E(self) -> np.ndarray:
        """Finds the next time step's body angular velocity (in the first Airplane's
        geometry axes, observed from the Earth frame) used to convect the wake.

        The wake convects over the interval ending at the next time step, so its
        apparent velocity from body rotation uses the next OperatingPoint's body rate
        rather than the current step's. The next OperatingPoint stores the body angular
        rate in the first Airplane's body axes, in degrees per second. This method
        transforms it to the first Airplane's geometry axes and converts it to radians
        per second for the omega cross r calculation in _apply_body_rate.

        :return: A (3,) ndarray of floats representing the next time step's body angular
            velocity (in the first Airplane's geometry axes, observed from the Earth
            frame). Its units are in radians per second.
        """
        next_operating_point = self._operating_point_at(self._current_step + 1)
        omegas_GP1__E = next_operating_point.omegas_BP1__E * _BP1_TO_GP1_FLIP
        return cast(np.ndarray, np.deg2rad(omegas_GP1__E))

    def freeze_substep(
        self, next_steady_problem: problems.SteadyProblem | None
    ) -> None:
        """Freezes the data the sub-iteration reuses across its trials.

        Called once at the start of a strongly coupled free-flight step solve, before
        any trial. Stores the transient next-step SteadyProblem (built by the caller
        over a scratch copy of the prescribed next-step Airplane) and the next step
        index, which redirect the inherited geometry and wake reads to the scratch copy.
        Precomputes the induced (Biot-Savart) part of the wake transport velocity, which
        depends only on the current step's bound geometry, strengths, and wake and so is
        identical across the step's trials (skipped for a prescribed wake, which has no
        induced part), and snapshots the current and previous steps' solved bound ring
        vortex strengths. evaluate_trial_aero_loads reuses these, and restore_substep
        clears them. The solver's current step, Airplanes, and strengths must be those
        of the step being solved, as they are when the run loop invokes the step solve.

        The official wake build after the solve accepts reuses this wake grid induced
        velocity instead of recomputing it (the step's state it reads is the same), so
        its singularity events are logged here.

        It also saves a reference to every solver attribute, which together hold the
        solved state of the step being solved, so restore_substep can put that state
        back after the trials instead of solving the step again.

        :param next_steady_problem: The transient SteadyProblem for the next step, built
            over a scratch copy of the prescribed next-step Airplane. Its OperatingPoint
            is a placeholder; the trial OperatingPoint is supplied per trial to
            evaluate_trial_aero_loads. On the array path it is None: the trials mesh the
            next time step's recorded step parameters, so no Airplane is copied.
        :return: None
        """
        self._substep_next_step = self._current_step + 1
        self._substep_next_steady_problem = next_steady_problem
        self._substep_next_operating_point = None

        if self._prescribed_wake:
            self._substepStackVIndGridWrvp_GP1__E = None
        else:
            bound_singularity_counts = np.zeros(3, dtype=np.int64)
            wake_singularity_counts = np.zeros(3, dtype=np.int64)
            self._substepStackVIndGridWrvp_GP1__E = (
                self._calculate_wake_grid_induced_velocities(
                    bound_singularity_counts, wake_singularity_counts
                )
            )
            _functions.log_unexpected_singularity_counts(
                _logger,
                logging.DEBUG,
                "_populate_next_airplanes_wake_vortex_points (bound)",
                bound_singularity_counts,
            )
            _functions.log_unexpected_singularity_counts(
                _logger,
                logging.DEBUG,
                "_populate_next_airplanes_wake_vortex_points (wake)",
                wake_singularity_counts,
            )

        self._substep_gamma_n = self._current_bound_vortex_strengths.copy()
        self._substep_gamma_n_minus_1 = self._last_bound_vortex_strengths.copy()

        # Save this step's solved state. The run loop has just solved this step, so the
        # solver's attributes (its geometry stacks, influences, strengths, loads, and
        # the pointers to the current step, Airplanes, and OperatingPoint) describe it.
        # The trials will point those attributes at the next step's data. Each solve
        # replaces the attributes with newly created arrays rather than writing into the
        # existing ones, so the arrays saved here are left untouched by the trials, and
        # saving references (not copies) is enough. All of the solver's state lives in
        # __slots__, so collecting the slots of every class in the hierarchy covers it.
        # The transient _substep slots are skipped, since restore_substep manages them
        # itself.
        self._substep_saved_state = {
            name: getattr(self, name)
            for cls in type(self).__mro__
            for name in getattr(cls, "__slots__", ())
            if not name.startswith("_substep") and hasattr(self, name)
        }

    def evaluate_trial_aero_loads(
        self,
        trial_operating_point: operating_point.OperatingPoint,
        step: int,
    ) -> geometry.airplane.Airplane | None:
        """Evaluates the aerodynamic loads at a trial body state for the next time step.

        Called once per sub-iteration with the trial OperatingPoint for the next step.
        Installs that OperatingPoint as the one the inherited reads see for the next
        step, builds the trial bound geometry at the next step, ages the current step's
        wake into the next step using the frozen induced transport from freeze_substep
        plus the trial frame terms, then solves the circulation and loads at the next
        step. The geometry and wake are built over the scratch copy of the next-step
        Airplane that freeze_substep installed. Returns that copy, carrying the trial
        aerodynamic loads, which the caller assembles into the interval load.

        The wake build runs with the current step as the reference step: its wake is the
        one aged, and its snapshot bound ring vortex strengths set the newly shed row,
        so the current step's context and snapshot strengths are restored before it. The
        circulation and load solve then runs at the next step against the trial geometry
        and the freshly aged wake, with the snapshot strengths supplied as the previous
        strengths for the unsteady load term.

        :param trial_operating_point: The trial OperatingPoint for the next step at
            which to evaluate the aerodynamics.
        :param step: The current time step index (zero indexed). The trial body state is
            the next step's.
        :return: The scratch copy of the next-step Airplane, carrying the trial
            aerodynamic forces and moments the solver calculated at the next step, or
            None on the array path, which keeps the trial loads in the solver's
            StepLoads instead.
        """
        assert self._substep_gamma_n is not None

        next_step = step + 1
        self._substep_next_operating_point = trial_operating_point

        # G: build the trial bound ring vortex geometry at the next step. It reads the
        # scratch-copy Airplane and the trial OperatingPoint through the inherited
        # _get_steady_problem_at and _operating_point_at dispatch.
        self._initialize_panel_vortices_at(next_step)

        # P: age the current step's wake into the next step. The wake builder reads the
        # current step as the reference step, the next step's trial OperatingPoint for
        # the frame terms, and the snapshot bound strengths for the newly shed row, so
        # set the current step's context and restore the snapshot strengths first. The
        # induced transport is the frozen precompute from freeze_substep. Changed for
        # the arrays rewrite (step 7): the array path reads only the current time step's
        # OperatingPoint, not its Airplanes.
        self._current_step = step
        if self._use_arrays:
            self.current_operating_point = self._operating_point_at(step)
        else:
            current_problem = self._get_steady_problem_at(step)
            self.current_airplanes = current_problem.airplanes
            self.current_operating_point = current_problem.operating_point
        self._currentVInf_GP1__E = self.current_operating_point.vInf_GP1__E
        self._current_bound_vortex_strengths = self._substep_gamma_n
        self._populate_next_airplanes_wake_vortex_points(
            self._substepStackVIndGridWrvp_GP1__E
        )
        self._populate_next_airplanes_wake_vortices()

        # C and L: solve the circulation and loads at the next step against the trial
        # geometry and the freshly aged wake. The current Airplanes are the scratch copy
        # and the current OperatingPoint is the trial. The unsteady load term
        # differences the solved strengths against the snapshot strengths.
        self._current_step = next_step
        # Changed for the arrays rewrite (step 7): the array path doesn't read the
        # scratch copy, which it doesn't build.
        if not self._use_arrays:
            self.current_airplanes = self._get_steady_problem_at(next_step).airplanes
        self.current_operating_point = trial_operating_point
        self._currentVInf_GP1__E = self.current_operating_point.vInf_GP1__E
        self._last_bound_vortex_strengths = self._substep_gamma_n
        self._evaluate_step_aerodynamics()

        if self._use_arrays:
            return None
        return self.current_airplanes[0]

    def keep_final_trial(self) -> None:
        """Keeps the final coupling trial's solution for the next time step's official
        solve.

        Called by the problem's step solve after the sub-iteration converges, while the
        solver still holds the final trial's solution at the next time step, and before
        restore_substep. _evaluate_step_aerodynamics then uses the kept solution in
        place of solving the next time step again (see there). The kept arrays are
        references, not copies: each solve creates new arrays rather than writing into
        existing ones, so nothing changes them before they are used.

        :return: None
        """
        assert self._substep_next_step is not None
        assert self._current_step == self._substep_next_step

        # A trial at a time step before the first results step calculated no loads, so
        # there is nothing to reuse, and the next time step is solved as usual.
        if self._current_step < self.first_results_step:
            self._substep_final_trial = None
            return

        assert self._stepStackPanelForces_GP1 is not None
        assert self._stepStackPanelMoments_GP1_CgP1 is not None
        self._substep_final_trial = _FinalTrial(
            step=self._current_step,
            strengths=self._current_bound_vortex_strengths,
            grid_wing_wing_influences=self._currentGridWingWingInfluences__E,
            stack_freestream_wing_influences=(
                self._currentStackFreestreamWingInfluences__E
            ),
            stack_wake_wing_influences=self._currentStackWakeWingInfluences__E,
            stackPanelForces_GP1=self._stepStackPanelForces_GP1,
            stackPanelMoments_GP1_CgP1=self._stepStackPanelMoments_GP1_CgP1,
        )

    def restore_substep(self, step: int) -> None:
        """Restores the solver's current-step state after a strongly coupled step solve.

        Called once after the sub-iteration accepts, before the inherited run loop
        builds the official wake for the accepted next step. The trials overwrote the
        current step's solver working state (the geometry stacks, influences, and
        strengths) while evaluating the next step, so this puts back the state that
        freeze_substep saved before the trials, leaving the solver as if the step had
        just been solved. Clears the transient sub-iteration state, so the inherited
        reads revert to the canonical committed SteadyProblems, except for the frozen
        wake grid induced velocity, which the official wake build takes over (see
        _populate_next_airplanes_wake).

        :param step: The current time step index (zero indexed).
        :return: None
        """
        assert self._substep_saved_state is not None

        # Changed for speed: this method used to solve the current step a second time
        # (the commented out block below) only to rebuild the solver state the trials
        # overwrote. That state is identical to the one freeze_substep saved before the
        # trials, so it is now put back directly, which skips a full solve (geometry
        # collapse, influence matrices, linear solve, and loads) every free step.
        # assert self._substep_gamma_n_minus_1 is not None
        # current_problem = self._get_steady_problem_at(step)
        # self._current_step = step
        # self.current_airplanes = current_problem.airplanes
        # self.current_operating_point = current_problem.operating_point
        # self._currentVInf_GP1__E = self.current_operating_point.vInf_GP1__E
        # self._last_bound_vortex_strengths = self._substep_gamma_n_minus_1
        # self._evaluate_step_aerodynamics()

        # Point every solver attribute back at the current step's saved state. This
        # includes the current step, Airplanes, OperatingPoint, freestream, and previous
        # strengths that the commented out block set by hand.
        for name, value in self._substep_saved_state.items():
            setattr(self, name, value)
        self._substep_saved_state = None

        # The saved state belongs to the step being solved, so the restored current step
        # must match the step the caller is restoring.
        assert self._current_step == step

        self._substep_next_step = None
        self._substep_next_steady_problem = None
        self._substep_next_operating_point = None
        # Changed for speed: the frozen wake grid induced velocity is no longer cleared
        # here. The official wake build reuses it and clears it in
        # _populate_next_airplanes_wake, instead of recomputing the same array.
        # self._substepStackVIndGridWrvp_GP1__E = None
        self._substep_gamma_n = None
        self._substep_gamma_n_minus_1 = None

    def _evaluate_step_aerodynamics(self) -> None:
        """Evaluates the aerodynamics for the current time step, either with a full
        solve or by reusing the final coupling trial kept for this time step.

        The parent's full solve has two halves: first it builds this time step's
        geometry and wake (_prepare_step_geometry), then it calculates the influence
        coefficients, solves for the bound ring vortex strengths, and calculates the
        loads. This method takes one of two paths:

        Full solve (the parent's method, both halves): on every time step without
        reuse_final_trial, on held time steps, on the first free flight time step, after
        a coupling that reached k_max, and inside the coupling's own trials.

        Reuse (the first half only): when the coupling of the previous time step kept
        its final trial for this time step. The geometry and wake are built exactly as
        in the full solve, and the strengths, influence coefficients, and per Panel
        loads are taken from the final trial instead of calculated. The trial was solved
        at a body state within the coupling tolerance of the accepted one, so its
        solution differs from the full solve's by about the same relative amount.

        The geometry is built rather than taken from the trial because part of it
        depends on the freestream (the trailing edge Panels' back ring vortex corners),
        and it must match the official wake, whose first row sits exactly on those
        corners, to the last bit. Otherwise a corner and a wake point a tiny distance
        apart induce a huge velocity on each other where they should cancel.

        :return: None
        """
        # Take the kept trial, if any, and clear it, so it can be used at most once.
        final_trial = self._substep_final_trial
        self._substep_final_trial = None

        # Reuse the kept trial only for the official solve of the time step it belongs
        # to. A trial kept for another time step is stale. The coupling's own trials run
        # while a substep is active (_substep_next_step is set), and they must always
        # solve in full, since each one is a new body state.
        reuse_final_trial = (
            final_trial is not None
            and final_trial.step == self._current_step
            and self._substep_next_step is None
        )

        if not reuse_final_trial:
            # Full solve: the parent builds the geometry and wake, then calculates the
            # influence coefficients, strengths, and loads.
            super()._evaluate_step_aerodynamics()
            return

        # Reuse (changed for speed, opt in): build only the geometry and wake, the first
        # half of the full solve, and take the rest from the final trial instead of
        # calculating it.
        assert final_trial is not None
        self._prepare_step_geometry()

        # In place of the influence coefficients and the linear solve.
        self._currentGridWingWingInfluences__E = final_trial.grid_wing_wing_influences
        self._currentStackFreestreamWingInfluences__E = (
            final_trial.stack_freestream_wing_influences
        )
        self._currentStackWakeWingInfluences__E = final_trial.stack_wake_wing_influences
        self._current_bound_vortex_strengths = final_trial.strengths

        # In place of the load calculation: write the trial's per Panel loads onto this
        # time step's official Airplanes and Panels through the same hook the load
        # calculation ends with. It finds the Airplanes' total loads and coefficients
        # with this time step's OperatingPoint (and, in the aeroelastic solver, the
        # moments about each strip's leading edge point from this time step's official
        # Panels, which the structural solve reads).
        self._process_panel_loads_hook(
            final_trial.stackPanelForces_GP1, final_trial.stackPanelMoments_GP1_CgP1
        )

    def _process_panel_loads_hook(
        self,
        forces_GP1: np.ndarray,
        moments_GP1_CgP1: np.ndarray,
    ) -> None:
        """Records the per Panel loads, then processes them as the parent does.

        keep_final_trial reads the recorded loads of the final coupling trial. The
        arrays are newly created by each load calculation and never written into
        afterwards, so keeping references is enough.

        :param forces_GP1: A (N,3) ndarray of floats representing the forces (in the
            first Airplane's geometry axes) on every Panel at the current time step. The
            units are in Newtons.
        :param moments_GP1_CgP1: A (N,3) ndarray of floats representing the moments (in
            the first Airplane's geometry axes, relative to the first Airplane's CG) on
            every Panel at the current time step. The units are in Newton-meters.
        :return: None
        """
        self._stepStackPanelForces_GP1 = forces_GP1
        self._stepStackPanelMoments_GP1_CgP1 = moments_GP1_CgP1
        super()._process_panel_loads_hook(forces_GP1, moments_GP1_CgP1)

    def _populate_next_airplanes_wake(self) -> None:
        """Updates the next time step's Airplanes' wakes.

        After an accepted strongly coupled step, reuses the wake grid induced velocity
        that freeze_substep computed, since this official wake build ages the same
        step's wake from the same restored state, and then clears it so a later step
        computes its own. On a prescribed step there is no frozen velocity (it is None),
        so the inherited wake build computes it as usual.

        :return: None
        """
        # The frozen array is only the induced (Biot-Savart) part of the wake transport
        # velocity, which freeze_substep computed from this time step's solved wake,
        # bound geometry, and strengths. The trials changed the body state, but not this
        # time step's state, and restore_substep has put the solver back to it, so the
        # array is exactly what the wake build would compute here. The parts that depend
        # on the body state (the freestream and the apparent velocity from body
        # rotation) are not in it: the wake build adds them fresh from the accepted next
        # step OperatingPoint.
        stackVIndGridWrvp_GP1__E = self._substepStackVIndGridWrvp_GP1__E
        self._substepStackVIndGridWrvp_GP1__E = None

        # Populate the locations of the next time step's Airplanes' wake ring vortex
        # points, reusing the frozen induced velocity when there is one.
        self._populate_next_airplanes_wake_vortex_points(stackVIndGridWrvp_GP1__E)

        # Populate the locations of the next time step's Airplanes' wake ring vortices.
        self._populate_next_airplanes_wake_vortices()
