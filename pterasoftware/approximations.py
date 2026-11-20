"""Contains the Approximations class, the opt in options that make free flight runs
faster in exchange for small, measured changes to their results."""

from __future__ import annotations

import numpy as np

from . import _parameter_validation

# This is the coupling's relative convergence tolerance when no approximation is on. It
# must equal problems._SUBITERATION_RELATIVE_TOLERANCE, which a unit test checks. See
# docs/STRONG_COUPLING.md, Section 8.
_EXACT_COUPLING_RELATIVE_TOLERANCE = 1e-6

# These are the bounds on coupling_relative_tolerance. The lower bound is the exact
# tolerance, so an Approximations can only loosen it. The upper bound is the coarsest
# tolerance worth using (see _FAST_COARSE_COUPLING_RELATIVE_TOLERANCE below).
_MIN_COUPLING_RELATIVE_TOLERANCE = _EXACT_COUPLING_RELATIVE_TOLERANCE
_MAX_COUPLING_RELATIVE_TOLERANCE = 1e-4

# This is the coupling's relative convergence tolerance that fast() sets. Measured on
# the FAAV flap driven free flight example (2 held and 6 free flaps), with
# adaptive_relaxation on: about 10% to 12% faster than the exact run at hover, 1 m/s, 3
# m/s, and 6 m/s (at 3 m/s, 3.57 sub-iterations per free flight time step against 4.96
# exact), with the first free flap's mean lift within 0.0003% to 0.0099% of the exact
# run's, well below the drift a 1e-12 m/s change in the air speed already causes
# (0.0066% to 0.57%).
_FAST_COUPLING_RELATIVE_TOLERANCE = 1e-5

# This is the coupling's relative convergence tolerance that fast_coarse() sets.
# Measured the same way: about 14% to 17% faster than the exact run (at 3 m/s, 2.82
# sub-iterations per free flight time step against 4.96 exact). Its error is larger than
# 1e-5's: at 1 m/s the first free flap's mean lift moved by 0.33%, more than the 0.25%
# that a 1e-12 m/s change in the air speed causes, which is why it is in fast_coarse()
# and not in fast().
#
# Going coarser than 1e-4 isn't worth it, which is why it is also the upper bound. A
# free flight time step needs at least 2 sub-iterations (one to evaluate the loads and
# one to confirm convergence), and 1e-4 already brings the average down to about 2.8. At
# 3 m/s, 3e-4 gave 2.22 sub-iterations (20% faster than exact) and 1e-3 gave exactly 2.0
# on every time step (22% faster), the floor, so loosening the tolerance past 1e-4 can
# save at most a few more percent while the error keeps growing. At 6 m/s, where the
# free flight is least sensitive, the first free flap's mean lift moved by 0.004% at
# 1e-4, but by 0.085% at 3e-4 and 0.046% at 1e-3, about 10 times what a 1e-12 m/s change
# in the air speed causes (0.0066%).
_FAST_COARSE_COUPLING_RELATIVE_TOLERANCE = 1e-4

# reuse_final_trial: after a free flight time step's coupling converges, the next time
# step's official solve takes its bound ring vortex strengths and per Panel loads from
# the coupling's final trial instead of solving again (it still builds its own geometry
# and wake). Measured alone on the FAAV flap driven free flight example (2 held and 6
# free flaps): about 4% to 8% faster than the exact run at hover, 1 m/s, 3 m/s, and 6
# m/s (5% to 9% per free flight time step), with the held flaps identical, the free
# flight time steps' forces within 5e-8 of the exact run's peak force over the first
# free flight time steps at 3 m/s, and the first free flap's mean lift within 0.0002%,
# 0.056%, and 0.0001% of the exact run's at 3 m/s, 1 m/s, and 6 m/s (against 0.57%,
# 0.25%, and 0.0066% for a 1e-12 m/s change in the air speed). On top of the other
# options it saves another 3% to 8%.


class Approximations:
    """A class used to contain the opt in approximations of a free flight problem.

    By default every option is off, and a problem built with Approximations() (or with
    approximations=None) gives exactly the same results as one built before this class
    existed. Each option trades a small, measured change in the results for speed.
    fast() returns the vetted set, which stays within the acceptance rule below.
    fast_coarse() returns a coarser set that is faster but can exceed it, for runs where
    speed matters more than matching the exact run closely.

    Every option acts only on the strong coupling between the aerodynamics and the rigid
    body dynamics in free flight (see docs/STRONG_COUPLING.md). Each free flight time
    step iterates until the body state and the aerodynamic loads agree within a
    tolerance. These options change how that iteration starts, when it stops, and
    whether the next time step solves again at the accepted state, not the equations it
    solves, so the converged answer of each step is the same fixed point within the
    tolerance. Free flight without attitude control is chaotic, so any change to the
    path of the iteration grows over a run. The acceptance rule for every option is
    therefore that free flight drifts less than a 1e-12 m/s change in the air speed
    already makes it drift, and that no step reaches the sub-iteration cap (k_max).

    **The options, and what was measured for them:**

    Measured on the FAAV flap driven free flight example (2 held and 6 free flaps, at
    hover, 1 m/s, 3 m/s, and 6 m/s). Together, as fast() sets them, they made the runs
    12% to 19% faster (and the pterosaur example 17% faster), the held flaps were
    identical, and the first free flap's mean lift drifted by 0.0001%, 0.042%, and
    0.0006% at 3 m/s, 1 m/s, and 6 m/s, against 0.57%, 0.25%, and 0.0066% for the 1e-12
    m/s air speed change. No step reached k_max.

    adaptive_relaxation: Each free flight time step's first sub-iteration update uses a
    relaxation factor of 0.5 by default (docs/STRONG_COUPLING.md, Section 9), a
    conservative start for very light vehicles. The Aitken method then computes the
    factor for every later sub-iteration. With adaptive_relaxation, each free flight
    time step instead starts at the last Aitken factor of the previous time step, if
    that factor was within [0.1, 1.0], and at 0.5 otherwise (and on the first free
    flight time step). That last factor is the damping the iteration measured on this
    vehicle one time step earlier, so a heavy vehicle, whose factor settles near 1.0,
    stops throwing away half of its first update, while a light vehicle, whose factor
    settles lower, still starts low. Alone, it made the runs 6% to 8% faster.

    coupling_relative_tolerance: The relative tolerance of the sub-iteration's
    convergence test (docs/STRONG_COUPLING.md, Section 8), 1e-6 by default. fast() sets
    1e-5, which Section 8 names as the lever for light vehicles. Alone, it made the runs
    5% to 8% faster. fast_coarse() sets 1e-4, the coarsest allowed (see the module's
    tolerance constants for the measurements, and for why coarser doesn't pay).

    reuse_final_trial: Each free flight time step's coupling solves the aerodynamics at
    the next time step for every trial body state, and the run loop then solves the next
    time step again at the accepted body state, which differs from the final trial's by
    less than the coupling tolerance. With reuse_final_trial, that second solve builds
    its own geometry and wake as usual but takes the bound ring vortex strengths and the
    per Panel loads from the final trial, skipping the influence coefficients, the
    linear solve, and the load calculation. A time step whose coupling reached k_max is
    solved as usual. Alone, it made the free flight time steps 5% to 8% faster.
    """

    __slots__ = (
        "_adaptive_relaxation",
        "_coupling_relative_tolerance",
        "_reuse_final_trial",
    )

    def __init__(
        self,
        adaptive_relaxation: bool | np.bool = False,
        coupling_relative_tolerance: float | int = _EXACT_COUPLING_RELATIVE_TOLERANCE,
        reuse_final_trial: bool | np.bool = False,
    ) -> None:
        """The initialization method.

        :param adaptive_relaxation: Set this to True to start each free flight time
            step's sub-iteration at the previous time step's last Aitken relaxation
            factor, when that factor is within [0.1, 1.0], instead of always at 0.5. See
            the class docstring. It can be a bool or a numpy bool and will be converted
            internally to a bool. The default is False.
        :param coupling_relative_tolerance: The relative tolerance of the sub-
            iteration's convergence test. It must be a number (int or float) in the
            range [1e-6, 1e-4] and will be converted internally to a float. It is
            dimensionless. The default is 1e-6, the exact tolerance.
        :param reuse_final_trial: Set this to True to take each free flight time step's
            bound ring vortex strengths and loads from the previous time step's final
            coupling trial, when that coupling converged, instead of solving for them
            again. See the class docstring. It can be a bool or a numpy bool and will be
            converted internally to a bool. The default is False.
        :return: None
        """
        self._adaptive_relaxation = _parameter_validation.boolLike_return_bool(
            adaptive_relaxation, "adaptive_relaxation"
        )
        self._coupling_relative_tolerance = (
            _parameter_validation.number_in_range_return_float(
                coupling_relative_tolerance,
                "coupling_relative_tolerance",
                min_val=_MIN_COUPLING_RELATIVE_TOLERANCE,
                min_inclusive=True,
                max_val=_MAX_COUPLING_RELATIVE_TOLERANCE,
                max_inclusive=True,
            )
        )
        self._reuse_final_trial = _parameter_validation.boolLike_return_bool(
            reuse_final_trial, "reuse_final_trial"
        )

    @classmethod
    def fast(cls) -> Approximations:
        """Returns the vetted set of approximations: adaptive_relaxation on, a
        coupling_relative_tolerance of 1e-5, and reuse_final_trial on.

        :return: An Approximations with every vetted option on.
        """
        return cls(
            adaptive_relaxation=True,
            coupling_relative_tolerance=_FAST_COUPLING_RELATIVE_TOLERANCE,
            reuse_final_trial=True,
        )

    @classmethod
    def fast_coarse(cls) -> Approximations:
        """Returns a coarser set of approximations than fast(), for runs where speed
        matters more than matching the exact run closely, such as exploring designs or
        tuning a controller: adaptive_relaxation on, a coupling_relative_tolerance of
        1e-4, and reuse_final_trial on.

        It is about 17% to 24% faster than the exact run on the FAAV (20% on the
        pterosaur example), against fast()'s 12% to 19%, but its free flight drift can
        reach what a 1e-12 m/s change in the air speed causes (at 1 m/s, 0.24% in the
        first free flap's mean lift, and 0.33% without reuse_final_trial, against
        0.25%).

        :return: An Approximations with the coarse set of options on.
        """
        return cls(
            adaptive_relaxation=True,
            coupling_relative_tolerance=_FAST_COARSE_COUPLING_RELATIVE_TOLERANCE,
            reuse_final_trial=True,
        )

    # --- Immutable: read only properties ---
    @property
    def adaptive_relaxation(self) -> bool:
        return self._adaptive_relaxation

    @property
    def coupling_relative_tolerance(self) -> float:
        return self._coupling_relative_tolerance

    @property
    def reuse_final_trial(self) -> bool:
        return self._reuse_final_trial

    # --- Derived (Mutable): read only properties, computed on every access ---
    @property
    def any_enabled(self) -> bool:
        """Whether any option differs from the exact behavior.

        :return: True if any option is on, and False otherwise.
        """
        return (
            self._adaptive_relaxation
            or self._coupling_relative_tolerance != _EXACT_COUPLING_RELATIVE_TOLERANCE
            or self._reuse_final_trial
        )

    def describe(self) -> str:
        """Returns a one line description of the options that are on, for logging.

        :return: A str listing the options that differ from the exact behavior, or
            "none" if none do.
        """
        enabled = []
        if self._adaptive_relaxation:
            enabled.append("adaptive_relaxation")
        if self._coupling_relative_tolerance != _EXACT_COUPLING_RELATIVE_TOLERANCE:
            enabled.append(
                f"coupling_relative_tolerance={self._coupling_relative_tolerance:g}"
            )
        if self._reuse_final_trial:
            enabled.append("reuse_final_trial")
        return ", ".join(enabled) if enabled else "none"
