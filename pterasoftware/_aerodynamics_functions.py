"""Contains functions for calculating induced velocities."""

from __future__ import annotations

import ctypes
import functools
import glob
import math
import os
import subprocess
import sys
import warnings

import numba
import numpy as np
from numba import njit, prange

from . import _logging

_logger = _logging.get_logger("_aerodynamics_functions")

# Squire's parameter relates to the size of the vortex cores and the rate at which they
# grow. The value of this parameter is slightly controversial. It dramatically affects
# the stability of the result. I'm using this value, as cited for use in flapping-wing
# vehicles in "Role of Filament Strain in the Free-Vortex Modeling of Rotor Wakes"
# (Ananthan and Leishman, 2004). It is unitless.
_SQUIRE = 1.0e-4

# Lamb's constant relates to the size of the vortex cores and the rate at which they
# grow. The value of this parameter is well agreed upon, and published in "Extended
# Unsteady Vortex-Lattice Method for Insect Flapping Wings" (Nguyen et al., 2016). It is
# unitless.
_LAMB = 1.25643

# The local machine error is used to detect degenerate (zero length) line vortices.
_EPS = sys.float_info.epsilon

# The minimum core radius of a line vortex, as a fraction of its length. Every line
# vortex gets at least this core, so a zero initial core radius means the floor.
#
# The floor trades two errors against each other. First, a core biases legitimate
# evaluations, especially those close to the line vortex. The most critical legitimate
# close evaluation is between a bound line vortex and its own Panel's collocation point,
# where the distance is on the order of the line vortex's own length. There, a core that
# is a fraction f of the length scales the induced velocity by about 1 - f^2, a relative
# error of f^2. Second, a core bounds the spurious velocity at a point that should lie
# on the line vortex, such as a load evaluation point at a leg's center. Rounding leaves
# such a point about _EPS times its distance from the reference point off the line
# vortex, and with a core the velocity there is about strength * offset / (2 * pi *
# core^2). Relative to a typical near field velocity of strength / (2 * pi * length),
# that is _EPS / f^2 when the point's distance from the reference point is about a
# length. Growing f shrinks the spurious error and grows the bias, and the two are equal
# at f = _EPS^(1 / 4), where both are about _EPS^(1 / 2).
#
# Two ratios were taken as one above. Evaluations closer than a length, such as a
# collocation point half a Panel chord from a long spanwise leg, raise the bias by the
# inverse square of their distance over the length. Reference points farther than a
# length from the line vortex raise the spurious error linearly in that distance over
# the length. Across all plausible ranges of both ratios, these cause relative errors
# much lower than the solvers' own accuracies, so the fraction is not adjusted for them.
_CORE_FRACTION = _EPS**0.25

# Pre compute 4 * pi and 4.0 * _LAMB as they used repeatedly.
_FOUR_PI = 4.0 * math.pi
_FOUR_LAMB = 4.0 * _LAMB

# The smallest number of evaluations worth handing one thread of a parallel kernel
# launch, where one evaluation is computing one line vortex's induced velocity at one
# point. Sized so a thread's share of the work takes several times longer than the cost
# of rallying the thread pool: waking its threads, splitting the work, and waiting for
# the slowest member to finish. A launch too small to clear one grain would spend more
# time rallying than computing, so it runs on a single thread. The grain is expressed as
# work per thread rather than as a core count, so it transfers across machines of
# different sizes without per-machine calibration.
_GRAIN = 10_000

# These set the chunk size of _collapsed_velocities_from_ring_vortices_vectorized, which
# hands each thread whole chunks of consecutive points and vectorizes across a chunk's
# points. The chunk size never changes the results, since every point's sum is computed
# the same way in any chunk. It is chosen per launch so each thread gets about
# _CHUNKS_PER_THREAD chunks (enough to balance the threads' work), but never fewer than
# _MIN_CHUNK_SIZE points (a few vectors' worth, even with AVX-512's 8 doubles, so the
# vectorized loop isn't dominated by its leftover iterations) and never more than
# _MAX_CHUNK_SIZE points (beyond which larger chunks gain nothing). This adapts to the
# machine's thread count and the launch's size without per machine tuning. The chunk
# size is also rounded up to a multiple of _CHUNK_SIZE_MULTIPLE (see
# collapsed_velocities_from_ring_vortices), and both bounds are multiples of it.
_CHUNKS_PER_THREAD = 4
_MIN_CHUNK_SIZE = 16
_MAX_CHUNK_SIZE = 128

# The vectorized kernel computes a chunk's points in vector passes as wide as the CPU's
# vector registers hold doubles (8 with AVX-512, 4 with AVX2), and any points left over
# at the end of the chunk one at a time, which is several times slower per point.
# Rounding the chunk size up to a multiple of 8 leaves no points over on either width.
# For example, about 730 points on 6 threads gives a chunk size of 31 (three passes of 8
# and 7 points left over), which becomes 32. Like the chunk size itself, this never
# changes the results.
_CHUNK_SIZE_MULTIPLE = 8

# WARNING, READ BEFORE CHANGING: these two constants pack two integer counts into one
# float. The vectorized ring vortex kernel
# (_collapsed_velocities_from_ring_vortices_vectorized) counts, for every point, how
# many line vortices have that point exactly on their start vertex and how many have it
# exactly on their end vertex. These counts only feed the singularity diagnostics
# (singularity_counts[1] and [2]); they never change a velocity. To make the kernel
# faster, it keeps both counts in ONE float64 per point:
#
#     packed = start_count * 1 + end_count * _VERTEX_HIT_BASE
#
# and after its loops it gets the two counts back with
#
#     start_count = packed % _VERTEX_HIT_BASE
#     end_count = packed // _VERTEX_HIT_BASE
#
# This is exact, not approximate, as long as BOTH of these hold:
#   1. Each count stays below _VERTEX_HIT_BASE (2^26 = 67,108,864). Otherwise the start
#      count would spill into the end count's digits and both would be wrong.
#   2. The packed total stays below 2^53. A float64 stores every whole number up to 2^53
#      exactly, and adding whole numbers never rounds below that. With condition 1, the
#      total is below 2^26 * 2^26 + 2^26, far below 2^53.
# A point can get at most one hit per (leg, ring vortex) pair, so each count is at most
# 4 * (number of ring vortices). collapsed_velocities_from_ring_vortices therefore
# raises a ValueError before launching the kernel if a launch has
# _MAX_RING_VORTICES_PER_LAUNCH or more ring vortices (about 16.8 million). Real runs
# have thousands to tens of thousands, and in practice a point is on at most about 4
# start and 4 end vertices (one corner shared by 4 ring vortices), so neither limit is
# ever close. If something goes wrong here, only the logged singularity counts would be
# wrong, never the induced velocities. Before raising _VERTEX_HIT_BASE, recheck
# condition 2.
_VERTEX_HIT_BASE = 2**26
_MAX_RING_VORTICES_PER_LAUNCH = _VERTEX_HIT_BASE // 4

# The same base as a float, so that the kernel's packed count update stays entirely in
# float64 arithmetic (mixing in an integer would undo the speed up). 2^26 is exactly
# representable as a float64.
_VERTEX_HIT_BASE_FLOAT = float(_VERTEX_HIT_BASE)


@functools.cache
def _physical_core_count() -> int:
    """Returns the number of physical CPU cores on this machine.

    A CPU with hyperthreading (simultaneous multithreading) runs two hardware threads on
    each physical core, and the operating system reports each as a logical processor, so
    os.cpu_count() counts them twice. _ceiling uses the physical count, for the reason
    its docstring gives.

    Python has no portable way to count physical cores, so this tries each operating
    system's own source in turn, and falls back to os.cpu_count() (the logical count,
    which is the ceiling's original behavior) if none works. psutil is not used, because
    Ptera Software doesn't depend on it. The count can't change while the process runs,
    so it is found once and cached.

    :return: A positive int representing the number of physical cores, never more than
        the number of logical processors.
    """
    logical_count = os.cpu_count() or 1
    for counter in (
        _physical_core_count_linux,
        _physical_core_count_macos,
        _physical_core_count_windows,
    ):
        try:
            count = counter()
        except Exception:
            # Any failure (a missing file, a missing system call, an unexpected format)
            # just moves on to the next source, so counting cores can never stop a
            # simulation. The last resort is the logical count, the original behavior.
            continue
        if count is not None and count > 0:
            return min(count, logical_count)
    return logical_count


def _physical_core_count_linux() -> int | None:
    """Returns the number of physical cores from Linux's sysfs CPU topology, or None
    when it isn't available.

    Each logical processor's topology directory names its physical package (socket) and
    its core within the package. Hyperthreads on one core share both, so the count of
    distinct pairs is the physical core count.

    :return: The number of physical cores, or None.
    """
    if not sys.platform.startswith("linux"):
        return None
    cores = set()
    for topology in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
        with open(os.path.join(topology, "physical_package_id")) as package_file:
            package_id = package_file.read().strip()
        with open(os.path.join(topology, "core_id")) as core_file:
            core_id = core_file.read().strip()
        cores.add((package_id, core_id))
    return len(cores) or None


def _physical_core_count_macos() -> int | None:
    """Returns the number of physical cores from macOS's hw.physicalcpu sysctl, or None
    when it isn't available.

    :return: The number of physical cores, or None.
    """
    if sys.platform != "darwin":
        return None
    output = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout
    return int(output.strip())


def _physical_core_count_windows() -> int | None:
    """Returns the number of physical cores from Windows'
    GetLogicalProcessorInformationEx, or None when it isn't available.

    Asked for processor core relationships only (RelationProcessorCore, 0), the call
    fills a buffer with one variable size record per physical core. Each record starts
    with its relationship type and then its size in bytes (two 32 bit integers), so the
    records are counted by stepping through the buffer by each one's size.

    :return: The number of physical cores, or None.
    """
    if sys.platform != "win32":
        return None
    # ctypes.WinDLL exists only on Windows, so it is looked up by name, which keeps type
    # checkers on other platforms from flagging it.
    get_information = getattr(ctypes, "WinDLL")(
        "kernel32"
    ).GetLogicalProcessorInformationEx
    relation_processor_core = 0
    buffer_length = ctypes.c_uint32(0)

    # The first call, with no buffer, fails but reports the buffer size it needs.
    get_information(relation_processor_core, None, ctypes.byref(buffer_length))
    if buffer_length.value == 0:
        return None
    buffer = ctypes.create_string_buffer(buffer_length.value)
    if not get_information(
        relation_processor_core, buffer, ctypes.byref(buffer_length)
    ):
        return None

    count = 0
    offset = 0
    while offset < buffer_length.value:
        record_size = int.from_bytes(buffer.raw[offset + 4 : offset + 8], "little")
        if record_size == 0:
            break
        count += 1
        offset += record_size
    return count


def _ceiling() -> int:
    """Returns the largest share of the Numba thread pool that a kernel launch may use.

    The share is three quarters of the smaller of the pool's width and the machine's
    physical core count, rounded down, never below 1.

    Why three quarters: a team that claims every core stalls on its closing barrier
    whenever anything else on the machine wants CPU time, because the scheduler delays
    one member and the barrier waits for that laggard. The main Python thread, which
    runs everything outside the kernels, and the operating system's own work need cores
    too. Three quarters leaves them that slack, and it costs nothing where it binds,
    having measured as the fastest width in every condition tested.

    Why physical cores rather than logical processors: with hyperthreading, a physical
    core's two hardware threads share all of its execution units (its vector multiply
    and add units, its load and store units, and its caches). A second thread on a core
    only gets work done while the first one is stalled, for example waiting on memory.
    The vectorized ring vortex kernel keeps a core's vector units busy almost all the
    time, with about 50 multiplies, adds, and memory operations per point and line
    vortex and few stalls, so a second thread on the same core mostly competes with the
    first for those units, and adds to the synchronization cost. On this project's
    development machine (8 physical cores, 16 logical processors), a cap of 6 kernel
    threads ran the free flight benchmark about 10% faster than the old cap of 12 (three
    quarters of the logical processors), and 8 ran slower than 6. On a CPU without
    hyperthreading, the two counts are equal and the ceiling is unchanged.

    The ceiling applies uniformly, including to thread masks set at or above it. Masks
    below the ceiling are honored exactly, so a batch of single thread simulations runs
    exactly as before. The thread count never changes the results, since each point's
    velocity is summed by one thread in a fixed order.

    The pool's width is read here rather than at import, because a module-level read
    would do arithmetic on a Numba attribute while the module is being imported, which
    fails wherever Numba is not fully importable (Sphinx's autodoc mocks it, for one).
    Reading it is cheap and, unlike numba.get_num_threads, does not launch the thread
    pool.

    :return: A positive int representing the most threads a single kernel launch may
        use.
    """
    # Numba populates its config module's attributes dynamically, so mypy sees this one
    # as untyped. Convert it explicitly rather than returning an untyped value.
    pool_width = int(numba.config.NUMBA_NUM_THREADS)

    # Changed for speed: the ceiling used to be three quarters of the pool's width alone
    # (the commented out line below), which by default is the number of logical
    # processors. It now uses the smaller of the pool's width and the physical core
    # count, so hyperthreads that would compete for the same core's vector units aren't
    # counted twice.
    # octowrap: off
    # return max((3 * pool_width) // 4, 1)
    # octowrap: on
    return max((3 * min(pool_width, _physical_core_count())) // 4, 1)


def _threads_for_launch(num_points: int, num_vortices: int) -> int:
    """Returns the work-proportional thread count for a parallel kernel launch.

    :param num_points: A non negative int representing the number of points at which the
        launch computes induced velocities.
    :param num_vortices: A non negative int representing the number of line vortices
        whose induced velocities the launch computes at each point.
    :return: A positive int representing the number of threads the launch's work
        justifies: one thread per whole grain of evaluations, floored at 1 and capped at
        the launch's point count.
    """
    evaluations = num_points * num_vortices

    # The kernels parallelize over points alone, summing each point's vortices in a
    # serial inner loop, so a launch can never use more threads than it has points.
    # Handing it more would leave the surplus threads with no iteration to run while
    # they still pay the parallel region's entry and its closing barrier, which is the
    # very overhead this dispatch exists to avoid. The cap binds when a launch carries
    # more than one grain of vortices and fewer points than the ceiling, as an unsteady
    # run's wake influences do on a small mesh once the wake grows, and as streamline
    # launches do generally.
    return min(max(evaluations // _GRAIN, 1), max(num_points, 1))


def report_thread_settings() -> None:
    """Reports the thread dispatch settings that a solver run's parallel kernel launches
    will operate under.

    Logged at the debug level: the active Numba threading layer, the Numba thread pool's
    width, the current thread mask, the kernel thread ceiling, and the upper bound those
    put on a launch's thread count. A launch whose work falls below the grain uses fewer
    threads than that bound.

    Raises a UserWarning, rather than logging one, if Numba is running on its workqueue
    threading layer. That layer slows every simulation, users land on it without being
    told, and the fix is a change to the user's environment rather than to anything
    Ptera Software can do, which is the case the warnings module exists for. The
    warnings module also suppresses the repeats itself, which matters because a trim or
    convergence analysis runs the solver many times over and Numba fixes the threading
    layer for the life of the process.

    :return: None
    """
    # Read the thread mask before the threading layer. Reading the mask launches the
    # thread pool if it isn't already running, and launching the pool initializes the
    # threading layer, so the layer is only queryable afterwards. The records below are
    # emitted in the opposite order, which the reads do not constrain.
    external_cap = numba.get_num_threads()
    threading_layer = numba.threading_layer()
    ceiling = _ceiling()
    kernel_cap = min(external_cap, ceiling)

    if threading_layer == "workqueue":
        warnings.warn(
            "Numba is running on its workqueue threading layer, which it falls "
            "back to when neither TBB nor OpenMP can be loaded. That layer charges "
            "a large fixed cost to every parallel kernel launch, no matter how "
            "small the launch or how few threads it uses, which slows every "
            "simulation. Install tbb, or a system OpenMP runtime, to let Numba use "
            "a faster layer. Numba also falls back to workqueue when an installed "
            "tbb cannot be loaded, so having tbb installed does not rule this out."
        )

    _logger.debug(_logging.indent() + f"Numba threading layer: {threading_layer}")
    _logger.debug(
        _logging.indent()
        + f"Numba thread pool width: {numba.config.NUMBA_NUM_THREADS}, thread mask: "
        + f"{external_cap}, physical cores: {_physical_core_count()}, ceiling: "
        + f"{ceiling} (three quarters of the smaller of the pool and the physical "
        + "cores)"
    )
    _logger.debug(
        _logging.indent()
        + f"Kernel launches will use at most {kernel_cap} threads, and fewer when a "
        + "launch's work is below the grain"
    )


def collapsed_velocities_from_ring_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of ring vortices and
    finds the cumulative induced velocity at every point.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M ring vortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M ring
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,4) ndarray of floats representing the initial core radii of the M
        ring vortices' right, front, left, and back legs. Pass zeros for bound legs,
        which then take the kernels' numerical floor. For wake legs, and for the
        trailing edge bound ring vortices' back legs that coincide with the first wake
        row's front legs, a reasonable value based on results from Ramasamy and Leishman
        (2007) is 3.0% the chord length of the parent Wing. The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across all four legs.
    :param ages: For bound ring vortices, this must be None. For ring vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M ring vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    # Changed for speed: this used to launch _collapsed_velocities_from_line_vortices
    # four times, once per leg, and add each leg's result array (the commented out code
    # below). One launch of _collapsed_velocities_from_ring_vortices_vectorized now
    # handles all four legs, and computes several points per vector instruction. Every
    # point still adds each leg's vortices in index order and then the four legs in the
    # same order (right, front, left, back), so the results are bit-identical. See that
    # function's docstring for how.
    #
    # octowrap: off
    # listStackSlvp_GP1_CgP1 = [
    #     stackBrrvp_GP1_CgP1,
    #     stackFrrvp_GP1_CgP1,
    #     stackFlrvp_GP1_CgP1,
    #     stackBlrvp_GP1_CgP1,
    # ]
    # listStackElvp_GP1_CgP1 = [
    #     stackFrrvp_GP1_CgP1,
    #     stackFlrvp_GP1_CgP1,
    #     stackBlrvp_GP1_CgP1,
    #     stackBrrvp_GP1_CgP1,
    # ]
    # list_r_c0s = [np.ascontiguousarray(r_c0s[:, i]) for i in range(4)]
    # stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3), dtype=float)
    # ...
    # try:
    #     for i in range(4):
    #         stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
    #             stackP_GP1_CgP1=stackP_GP1_CgP1,
    #             stackSlvp_GP1_CgP1=listStackSlvp_GP1_CgP1[i],
    #             stackElvp_GP1_CgP1=listStackElvp_GP1_CgP1[i],
    #             strengths=strengths,
    #             r_c0s=list_r_c0s[i],
    #             singularity_counts=singularity_counts,
    #             ages=ages,
    #             nu=nu,
    #         )
    # finally:
    #     numba.set_num_threads(external_cap)
    # octowrap: on

    num_points = stackP_GP1_CgP1.shape[0]
    num_vortices = strengths.shape[0]

    # Guard for the packed vertex hit counts. The vectorized kernel keeps each point's
    # start vertex and end vertex hit counts in one float64, which is exact only while
    # each count stays below _VERTEX_HIT_BASE. Each count is at most 4 * num_vortices,
    # so refuse any launch large enough to break that, instead of silently logging wrong
    # singularity counts. See the comment at _VERTEX_HIT_BASE. Real runs are about a
    # thousand times below this limit.
    if num_vortices >= _MAX_RING_VORTICES_PER_LAUNCH:
        raise ValueError(
            f"collapsed_velocities_from_ring_vortices got {num_vortices} ring "
            f"vortices, but it supports at most {_MAX_RING_VORTICES_PER_LAUNCH - 1} per "
            "call, because its kernel packs two singularity counts into one float64 "
            "(see _VERTEX_HIT_BASE in _aerodynamics_functions.py)."
        )

    # Bound ring vortices have no ages. The line vortex kernel treated None as every age
    # being 0.0, so pass that explicitly.
    if ages is None:
        ages = np.zeros(num_vortices, dtype=float)

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launch finishes.
    external_cap = numba.get_num_threads()
    num_threads = min(
        _threads_for_launch(num_points, num_vortices),
        external_cap,
        _ceiling(),
    )
    numba.set_num_threads(num_threads)

    # Choose the chunk size so each thread gets about _CHUNKS_PER_THREAD chunks, within
    # the bounds the constants' comment explains. It only affects speed.
    #
    # Changed for speed: the chunk size is now also rounded up to a multiple of
    # _CHUNK_SIZE_MULTIPLE, so no chunk ends in points computed one at a time outside
    # the vector passes (see _CHUNK_SIZE_MULTIPLE). Each point's operations and their
    # order are the same on either path, so the results are bit-identical. For the
    # loads' launches (about 730 points on 6 threads, a chunk size of 31 before), this
    # made the kernel about 24% faster. The bounds are multiples of
    # _CHUNK_SIZE_MULTIPLE, so rounding before clamping keeps the result a multiple too.
    # octowrap: off
    # chunk_size = min(
    #     _MAX_CHUNK_SIZE,
    #     max(_MIN_CHUNK_SIZE, -(-num_points // (_CHUNKS_PER_THREAD * num_threads))),
    # )
    # octowrap: on
    balanced_chunk_size = -(-num_points // (_CHUNKS_PER_THREAD * num_threads))
    rounded_chunk_size = (
        -(-balanced_chunk_size // _CHUNK_SIZE_MULTIPLE) * _CHUNK_SIZE_MULTIPLE
    )
    chunk_size = min(_MAX_CHUNK_SIZE, max(_MIN_CHUNK_SIZE, rounded_chunk_size))

    # Get the velocity induced by every leg of every ring vortex (in the first
    # Airplane's geometry axes, observed from the Earth frame).
    try:
        stackVInd_GP1__E: np.ndarray = (
            _collapsed_velocities_from_ring_vortices_vectorized(
                stackP_GP1_CgP1=np.ascontiguousarray(stackP_GP1_CgP1, dtype=float),
                stackBrrvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
                strengths=strengths,
                r_c0s=r_c0s,
                singularity_counts=singularity_counts,
                ages=ages,
                nu=nu,
                chunk_size=chunk_size,
            )
        )
    finally:
        numba.set_num_threads(external_cap)
    return stackVInd_GP1__E


def collapsed_velocities_from_ring_vortices_chordwise_segments(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of ring vortices and
    finds the cumulative induced velocity at every point due to the ring vortices' left
    and right line vortex legs.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M ring vortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M ring
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,4) ndarray of floats representing the initial core radii of the M
        ring vortices' right, front, left, and back legs. Pass zeros for bound legs,
        which then take the kernels' numerical floor. For wake legs, and for the
        trailing edge bound ring vortices' back legs that coincide with the first wake
        row's front legs, a reasonable value based on results from Ramasamy and Leishman
        (2007) is 3.0% the chord length of the parent Wing. The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across both legs.
    :param ages: For bound ring vortices, this must be None. For ring vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M ring vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame) due to the M ring vortices' left and right line vortex legs. The units
        are in meters per second.
    """
    listStackSlvp_GP1_CgP1 = [
        stackBrrvp_GP1_CgP1,
        stackFlrvp_GP1_CgP1,
    ]
    listStackElvp_GP1_CgP1 = [
        stackFrrvp_GP1_CgP1,
        stackBlrvp_GP1_CgP1,
    ]

    # The right and left legs' initial core radii are in the first and third columns.
    list_r_c0s = [np.ascontiguousarray(r_c0s[:, i]) for i in (0, 2)]

    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3), dtype=float)

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launches finish.
    external_cap = numba.get_num_threads()
    numba.set_num_threads(
        min(
            _threads_for_launch(stackP_GP1_CgP1.shape[0], strengths.shape[0]),
            external_cap,
            _ceiling(),
        )
    )

    # Get the velocity induced by the left and right legs of each ring vortex (in the
    # first Airplane's geometry axes, observed from the Earth frame).
    try:
        for i in range(2):
            stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
                stackP_GP1_CgP1=stackP_GP1_CgP1,
                stackSlvp_GP1_CgP1=listStackSlvp_GP1_CgP1[i],
                stackElvp_GP1_CgP1=listStackElvp_GP1_CgP1[i],
                strengths=strengths,
                r_c0s=list_r_c0s[i],
                singularity_counts=singularity_counts,
                ages=ages,
                nu=nu,
            )
    finally:
        numba.set_num_threads(external_cap)
    return stackVInd_GP1__E


def collapsed_velocities_from_line_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of line vortices and
    finds the cumulative induced velocity at every point due to the line vortices.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackSlvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M line vortices' starting vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackElvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M line vortices' ending vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M line
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        line vortices. Pass zeros for bound line vortices, which then take the kernels'
        numerical floor. For wake line vortices, a reasonable value based on results
        from Ramasamy and Leishman (2007) is 3.0% the chord length of the parent Wing.
        The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across calls.
    :param ages: For bound line vortices, this must be None. For line vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M line vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame) due to the M line vortices. The units are in meters per second.
    """
    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3), dtype=float)

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launch finishes.
    external_cap = numba.get_num_threads()
    numba.set_num_threads(
        min(
            _threads_for_launch(stackP_GP1_CgP1.shape[0], strengths.shape[0]),
            external_cap,
            _ceiling(),
        )
    )

    # Get the velocity induced by each line vortex (in the first Airplane's geometry
    # axes, observed from the Earth frame).
    try:
        stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
            stackP_GP1_CgP1=stackP_GP1_CgP1,
            stackSlvp_GP1_CgP1=stackSlvp_GP1_CgP1,
            stackElvp_GP1_CgP1=stackElvp_GP1_CgP1,
            strengths=strengths,
            r_c0s=r_c0s,
            singularity_counts=singularity_counts,
            ages=ages,
            nu=nu,
        )
    finally:
        numba.set_num_threads(external_cap)
    return stackVInd_GP1__E


def expanded_velocities_from_ring_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of ring vortices and
    finds the induced velocity at every point due to each ring vortex.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M ring vortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M ring
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,4) ndarray of floats representing the initial core radii of the M
        ring vortices' right, front, left, and back legs. Pass zeros for bound legs,
        which then take the kernels' numerical floor. For wake legs, and for the
        trailing edge bound ring vortices' back legs that coincide with the first wake
        row's front legs, a reasonable value based on results from Ramasamy and Leishman
        (2007) is 3.0% the chord length of the parent Wing. The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across all four legs.
    :param ages: For bound ring vortices, this must be None. For ring vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M ring vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M ring vortices. The units are in meters per second.
    """
    listStackSlvp_GP1_CgP1 = [
        stackBrrvp_GP1_CgP1,
        stackFrrvp_GP1_CgP1,
        stackFlrvp_GP1_CgP1,
        stackBlrvp_GP1_CgP1,
    ]
    listStackElvp_GP1_CgP1 = [
        stackFrrvp_GP1_CgP1,
        stackFlrvp_GP1_CgP1,
        stackBlrvp_GP1_CgP1,
        stackBrrvp_GP1_CgP1,
    ]

    list_r_c0s = [np.ascontiguousarray(r_c0s[:, i]) for i in range(4)]

    gridVInd_GP1__E = np.zeros(
        (stackP_GP1_CgP1.shape[0], strengths.shape[0], 3), dtype=float
    )

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launches finish.
    external_cap = numba.get_num_threads()
    numba.set_num_threads(
        min(
            _threads_for_launch(stackP_GP1_CgP1.shape[0], strengths.shape[0]),
            external_cap,
            _ceiling(),
        )
    )

    # Get the velocity induced by each leg of each ring vortex (in the first Airplane's
    # geometry axes, observed from the Earth frame).
    try:
        for i in range(4):
            gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
                stackP_GP1_CgP1=stackP_GP1_CgP1,
                stackSlvp_GP1_CgP1=listStackSlvp_GP1_CgP1[i],
                stackElvp_GP1_CgP1=listStackElvp_GP1_CgP1[i],
                strengths=strengths,
                r_c0s=list_r_c0s[i],
                singularity_counts=singularity_counts,
                ages=ages,
                nu=nu,
            )
    finally:
        numba.set_num_threads(external_cap)
    return gridVInd_GP1__E


# TODO: Remove the ability to specify horseshoe vortex ages, since they are never used
#  in unsteady simulations.
def collapsed_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrhvp_GP1_CgP1: np.ndarray,
    stackFrhvp_GP1_CgP1: np.ndarray,
    stackFlhvp_GP1_CgP1: np.ndarray,
    stackBlhvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of horseshoe vortices
    and finds the cumulative induced velocity at every point.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M horseshoe vortices' back right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' front left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackBlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' back left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        horseshoe vortices. The units are in meters squared per second.
    :param r_c0s: A (M,3) ndarray of floats representing the initial core radii of the M
        horseshoe vortices' right, front, and left legs. Pass zeros for bound front
        legs, which then take the kernels' numerical floor. For the right and left legs,
        which model the wake, a reasonable value based on results from Ramasamy and
        Leishman (2007) is 3.0% the chord length of the parent Wing. The units are in
        meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across all three legs.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    listStackSlvp_GP1_CgP1 = [
        stackBrhvp_GP1_CgP1,
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
    ]
    listStackElvp_GP1_CgP1 = [
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
        stackBlhvp_GP1_CgP1,
    ]

    list_r_c0s = [np.ascontiguousarray(r_c0s[:, i]) for i in range(3)]

    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3), dtype=float)

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launches finish.
    external_cap = numba.get_num_threads()
    numba.set_num_threads(
        min(
            _threads_for_launch(stackP_GP1_CgP1.shape[0], strengths.shape[0]),
            external_cap,
            _ceiling(),
        )
    )

    # Get the velocity induced by each leg of each horseshoe vortex (in the first
    # Airplane's geometry axes, observed from the Earth frame).
    try:
        for i in range(3):
            stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
                stackP_GP1_CgP1=stackP_GP1_CgP1,
                stackSlvp_GP1_CgP1=listStackSlvp_GP1_CgP1[i],
                stackElvp_GP1_CgP1=listStackElvp_GP1_CgP1[i],
                strengths=strengths,
                r_c0s=list_r_c0s[i],
                singularity_counts=singularity_counts,
                ages=None,
                nu=nu,
            )
    finally:
        numba.set_num_threads(external_cap)
    return stackVInd_GP1__E


# TODO: Remove the ability to specify horseshoe vortex ages, since they are never used
#  in unsteady simulations.
def expanded_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrhvp_GP1_CgP1: np.ndarray,
    stackFrhvp_GP1_CgP1: np.ndarray,
    stackFlhvp_GP1_CgP1: np.ndarray,
    stackBlhvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of horseshoe vortices
    and finds the induced velocity at every point due to each horseshoe vortex.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations. Each kernel
    launch runs with a work-proportional thread count, which never exceeds the launch's
    point count, the current Numba thread mask, or the module's kernel thread ceiling.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M horseshoe vortices' back right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' front left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackBlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M horseshoe vortices' back left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of M horseshoe
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,3) ndarray of floats representing the initial core radii of the M
        horseshoe vortices' right, front, and left legs. Pass zeros for bound front
        legs, which then take the kernels' numerical floor. For the right and left legs,
        which model the wake, a reasonable value based on results from Ramasamy and
        Leishman (2007) is 3.0% the chord length of the parent Wing. The units are in
        meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across all three legs.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M horseshoe vortices. The units are in meters per second.
    """
    listStackSlvp_GP1_CgP1 = [
        stackBrhvp_GP1_CgP1,
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
    ]
    listStackElvp_GP1_CgP1 = [
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
        stackBlhvp_GP1_CgP1,
    ]

    list_r_c0s = [np.ascontiguousarray(r_c0s[:, i]) for i in range(3)]

    gridVInd_GP1__E = np.zeros(
        (stackP_GP1_CgP1.shape[0], strengths.shape[0], 3), dtype=float
    )

    # Read the current thread mask so the dispatch honors any cap the user has set, and
    # restore it once the kernel launches finish.
    external_cap = numba.get_num_threads()
    numba.set_num_threads(
        min(
            _threads_for_launch(stackP_GP1_CgP1.shape[0], strengths.shape[0]),
            external_cap,
            _ceiling(),
        )
    )

    # Get the velocity induced by each leg of each horseshoe vortex (in the first
    # Airplane's geometry axes, observed from the Earth frame).
    try:
        for i in range(3):
            gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
                stackP_GP1_CgP1=stackP_GP1_CgP1,
                stackSlvp_GP1_CgP1=listStackSlvp_GP1_CgP1[i],
                stackElvp_GP1_CgP1=listStackElvp_GP1_CgP1[i],
                strengths=strengths,
                r_c0s=list_r_c0s[i],
                singularity_counts=singularity_counts,
                ages=None,
                nu=nu,
            )
    finally:
        numba.set_num_threads(external_cap)
    return gridVInd_GP1__E


@njit(cache=True, fastmath=False, parallel=True)
def _collapsed_velocities_from_line_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of line vortices and
    finds the cumulative induced velocity at every point.

    This function uses a modified version of the Biot-Savart law to create a smooth
    induced velocity decay based on a line vortex's core radius. The core radius grows
    from an initial value based on the line vortex's age.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    **Citation:**

    Core radius equation adapted from Eq. 3 of: "A Reynolds Number-Based Blade Tip
    Vortex Model"

    Authors: Manikandan Ramasamy and J. Gordon Leishman

    Biot-Savart equation adapted from: "Extended Unsteady Vortex-Lattice Method for
    Insect Flapping Wings"

    Authors: Anh Tuan Nguyen, Joong-Kwan Kim, Jong-Seob Han, and Jae-Hung Han

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackSlvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M line vortices' starting vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackElvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M line vortices' ending vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M line
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        line vortices. Pass zeros for bound line vortices, which then take the kernels'
        numerical floor. For wake line vortices, a reasonable value based on results
        from Ramasamy and Leishman (2007) is 3.0% the chord length of the parent Wing.
        The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across calls.
    :param ages: For bound line vortices, this must be None. For line vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M line vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Initialize an empty array, which we will fill with the induced velocities (in the
    # first Airplane's geometry axes, observed from the Earth frame).
    stackVInd_GP1__E = np.zeros((num_points, 3))

    # If the user didn't specify any ages, set the age of each line vortex to 0.0
    # seconds.
    if ages is None:
        ages = np.zeros(num_vortices)

    # Pre compute per line vortex quantities in a serial pass. This hoists invariant
    # work out of the parallel point loop and correctly counts degenerate filaments,
    # which are a per line vortex property, not per point.
    vortex_valid = np.empty(num_vortices, dtype=np.bool)
    vortex_c1 = np.empty(num_vortices)
    vortex_c2 = np.empty(num_vortices)

    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        # The r0_GP1 vector goes from the line vortex's start point to its end point (in
        # the first Airplane's geometry axes).
        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        # Find r0_GP1's length.
        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        # Skip degenerate filaments where the start and end points coincide.
        if r0 < _EPS:
            singularity_counts[0] += 1
            vortex_valid[vortex_id] = False
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        age = ages[vortex_id]

        # Clamp the initial core radius to the numerical floor.
        r_c0 = max(r_c0s[vortex_id], _CORE_FRACTION * r0)

        # Calculate the radius of the line vortex's core squared.
        r_c_sq = r_c0**2.0 + _FOUR_LAMB * (nu + _SQUIRE * abs(strength)) * age

        vortex_c1[vortex_id] = strength / _FOUR_PI
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Use per point singularity counts to avoid write races in the parallel loop. Index
    # mapping: [1] point on start vertex, [2] point on end vertex. Index [0] (degenerate
    # filament) is handled in the serial pre pass above.
    point_singularity_counts = np.zeros((num_points, 3), dtype=np.int64)

    # Numba does not annotate prange.__new__, so mypy cannot tell that calling prange
    # returns an iterable. Once this is fixed upstream, warn_unused_ignores will flag
    # the ignore below as unused.
    for point_id in prange(num_points):  # type: ignore[attr-defined]
        P_GP1_CgP1 = stackP_GP1_CgP1[point_id]

        for vortex_id in range(num_vortices):
            if not vortex_valid[vortex_id]:
                continue

            Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
            Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

            # The r0_GP1 vector goes from the line vortex's start point to its end point
            # (in the first Airplane's geometry axes).
            r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
            r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
            r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

            # The r1_GP1 vector goes from P_GP1_CgP1 to the line vortex's start point
            # (in the first Airplane's geometry axes).
            r1X_GP1 = Slvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r1Y_GP1 = Slvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r1Z_GP1 = Slvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r2_GP1 vector goes from P_GP1_CgP1 to the line vortex's end point (in
            # the first Airplane's geometry axes).
            r2X_GP1 = Elvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r2Y_GP1 = Elvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r2Z_GP1 = Elvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r3_GP1 vector is the cross product of r1_GP1 and r2_GP1 (in the first
            # Airplane's geometry axes). It is computed as r1_GP1 x r0_GP1, which is
            # equal because r2_GP1 = r1_GP1 + r0_GP1, but avoids cancellation when the
            # point is far from the line vortex relative to its length.
            r3X_GP1 = r1Y_GP1 * r0Z_GP1 - r1Z_GP1 * r0Y_GP1
            r3Y_GP1 = r1Z_GP1 * r0X_GP1 - r1X_GP1 * r0Z_GP1
            r3Z_GP1 = r1X_GP1 * r0Y_GP1 - r1Y_GP1 * r0X_GP1

            # Find the lengths of r1_GP1 and r2_GP1.
            r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
            r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)

            # Cache squared length of r3_GP1 as it is used in the c_4 calculation.
            r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0

            # Cache r1 * r2 as it is used twice in the c_4 calculation.
            r1_times_r2 = r1 * r2

            c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

            # The denominator is zero only if the point is on a vertex, as the core
            # radius is always positive. Skip these true singularities.
            denominator = r1_times_r2 * (r3_sq + vortex_c2[vortex_id])
            if denominator == 0.0:
                if r1 == 0.0:
                    point_singularity_counts[point_id, 1] += 1
                elif r2 == 0.0:
                    point_singularity_counts[point_id, 2] += 1
                continue

            # Find r1 * r2 - c_3 in the form that doesn't cancel for this angle. For c_3
            # >= 0.0 (theta <= 90 degrees), use the identity r1 * r2 - c_3 = r3_sq / (r1
            # * r2 + c_3). For c_3 < 0.0, the direct difference is safe.
            if c_3 >= 0.0:
                numerator = r3_sq / (r1_times_r2 + c_3)
            else:
                numerator = r1_times_r2 - c_3

            c_4 = vortex_c1[vortex_id] * (r1 + r2) * numerator / denominator
            stackVInd_GP1__E[point_id, 0] += c_4 * r3X_GP1
            stackVInd_GP1__E[point_id, 1] += c_4 * r3Y_GP1
            stackVInd_GP1__E[point_id, 2] += c_4 * r3Z_GP1

    # Aggregate per point singularity counts into the output array.
    for k in range(1, 3):
        for p in range(num_points):
            singularity_counts[k] += point_singularity_counts[p, k]

    return stackVInd_GP1__E


# error_model="numpy" makes a floating point division by zero give inf or NaN instead of
# raising. With the default "python" model, Numba checks every division, and those
# checks stop LLVM from vectorizing the inner loop and Numba from running the outer loop
# in parallel. The only division that can see a zero here is at a point exactly on a
# vertex, and its value is discarded by a select (see below), so the results are the
# same either way.
#
# Changed for speed: fastmath={"contract"} lets LLVM fuse each multiply and the add that
# uses it into one fused multiply add instruction, which rounds once instead of twice.
# Nothing is reordered, so each point still adds its vortices one at a time in index
# order, and the vectorization across points is kept. The kernel is limited by its total
# multiply, add, and memory work, so this makes it about 6% faster. The results now
# differ from the unfused kernel at round off level (the fused operations round once).
# octowrap: off
# @njit(cache=True, fastmath=False, parallel=True, error_model="numpy")
# octowrap: on
@njit(cache=True, fastmath={"contract"}, parallel=True, error_model="numpy")
def _collapsed_velocities_from_ring_vortices_vectorized(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray,
    nu: float,
    chunk_size: int,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of ring vortices and
    finds the cumulative induced velocity at every point, computing several points per
    vector instruction.

    This gives the same results, to round off, as calling the line vortex kernel
    (_collapsed_velocities_from_line_vortices) once per leg (right, front, left, back)
    and adding the four results, which is what collapsed_velocities_from_ring_vortices
    used to do. Without its fused multiply adds (see the decorator's comment) it gives
    the same results bit for bit. It exists for speed: it handles all four legs in one
    launch, and its innermost loop runs over a chunk of points, which LLVM vectorizes
    (on AVX-512, 8 points per instruction). Vectorizing across points rather than across
    vortices is what keeps each point's additions in order, since each point still adds
    its vortices one at a time in index order.

    Why the line vortex kernel is slower, even though Numba compiles it: the CPU's
    vector units can process several double precision numbers per instruction (8 with
    AVX-512), but that kernel computes one filament for one point per instruction, so
    most of each vector unit sits idle, and it runs at a few percent of the CPU's peak.
    LLVM, the compiler behind Numba, vectorizes a loop only if it can do so without
    changing the results, and two things in that kernel stop it. First, its inner loop
    runs over the vortices and adds into one sum per point, so vectorizing it would
    reorder the additions, which changes the rounding, and LLVM correctly refuses.
    Second, the loop has exits: a continue for each skipped pair, and the zero checks
    Numba's default error model puts before every division, which are possible exception
    exits. It also launches once per leg, so each ring vortex call starts the thread
    team four times, reads every point four times, and allocates four temporary arrays.

    This kernel turns the loop around: its innermost loop runs over points, which are
    independent of each other, so several points (8 with AVX-512) are computed per
    instruction with nothing reordered. The skips became selects, the division checks
    are turned off with error_model="numpy", and one launch handles all four legs.

    Each point's velocity is ((0.0 + right) + front) + left) + back, where each leg's
    sum adds that leg's vortices in index order starting from 0.0, exactly as the four
    launches and their additions did.

    The per vortex quantities come from the same serial pre-pass as the line vortex
    kernel's. Degenerate vortices are removed before the point loop (and counted)
    instead of skipped inside it, so the rest are still added in the same order.

    A point exactly on a vertex contributes exactly 0.0 instead of being skipped. A sum
    that starts at 0.0 never becomes -0.0 in round to nearest arithmetic, so adding 0.0
    to it never changes it. The skipped pair's own value (a 0 / 0) is replaced by a
    select, not multiplied by zero, so a NaN can't reach the sum.

    The two forms of the numerator are chosen per point by the same test, so each point
    gets the same form as before. Square roots and divisions are correctly rounded in
    IEEE arithmetic, so their vector forms give the same bits as their scalar forms. The
    only fastmath flag on is contract, which fuses multiplies with the adds that use
    them, but nothing is reordered. A fused multiply add with a 0.0 product (the
    selected out pairs) still leaves the sum unchanged, so the selects work as before.

    The chunk size only affects speed (see the module's chunk size constants).

    The singularity counts for points on a start vertex and on an end vertex are kept
    packed in one float64 per point (start + end * _VERTEX_HIT_BASE) inside the point
    loop, and unpacked once per point at the end. One float array costs much less in the
    vectorized loop than two integer arrays, and it is exact while M is below
    _MAX_RING_VORTICES_PER_LAUNCH, which collapsed_velocities_from_ring_vortices
    enforces. See the comment at _VERTEX_HIT_BASE. The velocities don't depend on this.

    :param stackP_GP1_CgP1: A (N,3) C contiguous ndarray of floats representing the
        positions of N points (in the first Airplane's geometry axes, relative to the
        first Airplane's CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M ring vortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters. M must be below
        _MAX_RING_VORTICES_PER_LAUNCH (see above).
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M ring vortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M ring
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,4) ndarray of floats representing the initial core radii of the M
        ring vortices' right, front, left, and back legs. The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across all four legs.
    :param ages: A (M,) ndarray of floats representing the ages of the M ring vortices
        in seconds. Pass zeros for bound ring vortices.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second.
    :param chunk_size: A positive int representing the number of consecutive points each
        parallel iteration handles.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    num_points = stackP_GP1_CgP1.shape[0]
    num_vortices = strengths.shape[0]

    # Pre compute each leg's per line vortex quantities in a serial pass, with the same
    # arithmetic as _collapsed_velocities_from_line_vortices's pre-pass. Each leg's
    # valid line vortices are stored compactly in index order in row leg of these (4,
    # M) arrays, and degenerate ones are counted and left out. Leg order: right (back
    # right to front right), front (front right to front left), left (front left to back
    # left), back (back left to back right).
    legStartsX = np.empty((4, num_vortices))
    legStartsY = np.empty((4, num_vortices))
    legStartsZ = np.empty((4, num_vortices))
    legEndsX = np.empty((4, num_vortices))
    legEndsY = np.empty((4, num_vortices))
    legEndsZ = np.empty((4, num_vortices))
    leg_c1 = np.empty((4, num_vortices))
    leg_c2 = np.empty((4, num_vortices))
    num_valid = np.zeros(4, dtype=np.int64)
    for leg in range(4):
        if leg == 0:
            stackSlvp_GP1_CgP1 = stackBrrvp_GP1_CgP1
            stackElvp_GP1_CgP1 = stackFrrvp_GP1_CgP1
        elif leg == 1:
            stackSlvp_GP1_CgP1 = stackFrrvp_GP1_CgP1
            stackElvp_GP1_CgP1 = stackFlrvp_GP1_CgP1
        elif leg == 2:
            stackSlvp_GP1_CgP1 = stackFlrvp_GP1_CgP1
            stackElvp_GP1_CgP1 = stackBlrvp_GP1_CgP1
        else:
            stackSlvp_GP1_CgP1 = stackBlrvp_GP1_CgP1
            stackElvp_GP1_CgP1 = stackBrrvp_GP1_CgP1
        count = 0
        for vortex_id in range(num_vortices):
            Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
            Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]
            r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
            r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
            r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]
            r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

            # Leave out degenerate filaments where the start and end points coincide.
            if r0 < _EPS:
                singularity_counts[0] += 1
                continue

            strength = strengths[vortex_id]
            age = ages[vortex_id]
            r_c0 = max(r_c0s[vortex_id, leg], _CORE_FRACTION * r0)
            r_c_sq = r_c0**2.0 + _FOUR_LAMB * (nu + _SQUIRE * abs(strength)) * age

            legStartsX[leg, count] = Slvp_GP1_CgP1[0]
            legStartsY[leg, count] = Slvp_GP1_CgP1[1]
            legStartsZ[leg, count] = Slvp_GP1_CgP1[2]
            legEndsX[leg, count] = Elvp_GP1_CgP1[0]
            legEndsY[leg, count] = Elvp_GP1_CgP1[1]
            legEndsZ[leg, count] = Elvp_GP1_CgP1[2]
            leg_c1[leg, count] = strength / _FOUR_PI
            leg_c2[leg, count] = r0**2.0 * r_c_sq
            count += 1
        num_valid[leg] = count

    # Split the points into their coordinates, so a chunk's coordinates are contiguous.
    pointsX = np.ascontiguousarray(stackP_GP1_CgP1[:, 0])
    pointsY = np.ascontiguousarray(stackP_GP1_CgP1[:, 1])
    pointsZ = np.ascontiguousarray(stackP_GP1_CgP1[:, 2])

    stackVInd_GP1__E = np.zeros((num_points, 3))

    # Per point singularity counts avoid write races in the parallel loop. Index
    # mapping: [1] point on start vertex, [2] point on end vertex.
    point_singularity_counts = np.zeros((num_points, 3), dtype=np.int64)

    num_chunks = (num_points + chunk_size - 1) // chunk_size
    for chunk_id in prange(num_chunks):
        start = chunk_id * chunk_size
        n = min(chunk_size, num_points - start)
        chunkX = pointsX[start : start + n]
        chunkY = pointsY[start : start + n]
        chunkZ = pointsZ[start : start + n]

        # Each point's total over the four legs, which starts at 0.0 as the line vortex
        # kernel's result array did.
        allLegsX = np.zeros(n)
        allLegsY = np.zeros(n)
        allLegsZ = np.zeros(n)

        # Changed for speed: the two per point integer count arrays below were replaced
        # by one float64 array, packed_vertex_hits, which holds both counts as
        # start_count + end_count * _VERTEX_HIT_BASE. Updating two int64 arrays in the
        # vectorized loop below cost about a third of the kernel's time (two more arrays
        # to load and store, and converting float compare results to integer lanes). The
        # packing is exact under the limits described at _VERTEX_HIT_BASE, and it never
        # affects the velocities.
        # octowrap: off
        # start_vertex_counts = np.zeros(n, dtype=np.int64)
        # end_vertex_counts = np.zeros(n, dtype=np.int64)
        # octowrap: on
        packed_vertex_hits = np.zeros(n)

        for leg in range(4):
            # Each point's sum over this leg's vortices, starting at 0.0.
            legX = np.zeros(n)
            legY = np.zeros(n)
            legZ = np.zeros(n)
            for v in range(num_valid[leg]):
                SX = legStartsX[leg, v]
                SY = legStartsY[leg, v]
                SZ = legStartsZ[leg, v]
                EX = legEndsX[leg, v]
                EY = legEndsY[leg, v]
                EZ = legEndsZ[leg, v]
                c1 = leg_c1[leg, v]
                c2 = leg_c2[leg, v]

                # The r0_GP1 vector goes from the line vortex's start point to its end
                # point (in the first Airplane's geometry axes).
                r0X_GP1 = EX - SX
                r0Y_GP1 = EY - SY
                r0Z_GP1 = EZ - SZ

                # This innermost loop over the chunk's points is the one LLVM
                # vectorizes. Its formulas are the line vortex kernel's, in the same
                # order.
                for p in range(n):
                    r1X_GP1 = SX - chunkX[p]
                    r1Y_GP1 = SY - chunkY[p]
                    r1Z_GP1 = SZ - chunkZ[p]
                    r2X_GP1 = EX - chunkX[p]
                    r2Y_GP1 = EY - chunkY[p]
                    r2Z_GP1 = EZ - chunkZ[p]

                    # r3_GP1 = r1_GP1 x r0_GP1, which equals r1_GP1 x r2_GP1 but avoids
                    # cancellation far from the line vortex.
                    r3X_GP1 = r1Y_GP1 * r0Z_GP1 - r1Z_GP1 * r0Y_GP1
                    r3Y_GP1 = r1Z_GP1 * r0X_GP1 - r1X_GP1 * r0Z_GP1
                    r3Z_GP1 = r1X_GP1 * r0Y_GP1 - r1Y_GP1 * r0X_GP1

                    r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
                    r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)
                    r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0
                    r1_times_r2 = r1 * r2
                    c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

                    # The denominator is zero only if the point is on a vertex. The line
                    # vortex kernel skipped these pairs. Here they contribute exactly
                    # 0.0 (see the docstring), and are counted.
                    denominator = r1_times_r2 * (r3_sq + c2)
                    on_vertex = denominator == 0.0

                    # r1 * r2 - c_3 in the form that doesn't cancel for this angle.
                    numerator = (
                        r3_sq / (r1_times_r2 + c_3) if c_3 >= 0.0 else r1_times_r2 - c_3
                    )
                    c_4 = 0.0 if on_vertex else c1 * (r1 + r2) * numerator / denominator

                    legX[p] += c_4 * r3X_GP1
                    legY[p] += c_4 * r3Y_GP1
                    legZ[p] += c_4 * r3Z_GP1

                    # Changed for speed: count this pair's vertex hit in the packed
                    # float instead of the two integer arrays (the old code is commented
                    # out below). The conditions are the old ones exactly:
                    #   - point on the start vertex (on_vertex and r1 == 0.0): add 1.0,
                    #     which counts one in the start count's digits;
                    #   - otherwise, point on the end vertex (on_vertex and r2 == 0.0,
                    #     and r1 != 0.0 because the first branch wasn't taken): add
                    #     _VERTEX_HIT_BASE_FLOAT (2^26), which counts one in the end
                    #     count's digits;
                    #   - otherwise: add 0.0, no hit.
                    # Every value added is a whole number and the total stays far below
                    # 2^53, so each addition is exact (see _VERTEX_HIT_BASE).
                    # octowrap: off
                    # start_vertex_counts[p] += 1 if (on_vertex and r1 == 0.0) else 0
                    # end_vertex_counts[p] += (
                    #     1 if (on_vertex and r1 != 0.0 and r2 == 0.0) else 0
                    # )
                    # octowrap: on
                    #
                    # Changed for speed again: the packed count update below used to
                    # test on_vertex in both branches (the commented out code just
                    # below). It now tests only r1 and r2, combined with a bitwise or
                    # (|), which LLVM turns into two vector compares and one blend,
                    # while the "and" chain compiled into a much longer mask and blend
                    # sequence. This made the kernel about 5% to 13% faster and the
                    # whole run about 4% faster, with bit-identical results.
                    #
                    # Why the counts are the same as before: if r1 == 0.0, then
                    # r1_times_r2 == 0.0, so denominator == 0.0 and on_vertex is True
                    # (the same holds for r2). So "on_vertex and r1 == 0.0" is the same
                    # test as "r1 == 0.0", and likewise for r2. The new update gives:
                    #   - r1 == 0.0: add 1.0 (a start vertex hit), as before;
                    #   - r1 != 0.0 and r2 == 0.0: add _VERTEX_HIT_BASE_FLOAT (an end
                    #     vertex hit), as before;
                    #   - neither: add 0.0 (no hit), as before.
                    # The one exception needs a non finite number: if r2 or (r3_sq + c2)
                    # were inf, then 0.0 * inf is NaN, on_vertex would be False, and the
                    # old code wouldn't count the hit while the new code would. That
                    # needs coordinates near 1e154 m or an overflowing core radius,
                    # which don't occur, and even then only the logged singularity count
                    # could differ. The velocities never depend on this update: c_4
                    # above still uses on_vertex.
                    # octowrap: off
                    # packed_vertex_hits[p] += (
                    #     1.0
                    #     if (on_vertex and r1 == 0.0)
                    #     else (
                    #         _VERTEX_HIT_BASE_FLOAT if (on_vertex and r2 == 0.0) else 0.0
                    #     )
                    # )
                    # octowrap: on
                    packed_vertex_hits[p] += (
                        (1.0 if r1 == 0.0 else _VERTEX_HIT_BASE_FLOAT)
                        if ((r1 == 0.0) | (r2 == 0.0))
                        else 0.0
                    )

            # Add this leg's sums to the totals, in leg order.
            for p in range(n):
                allLegsX[p] += legX[p]
                allLegsY[p] += legY[p]
                allLegsZ[p] += legZ[p]

        for p in range(n):
            stackVInd_GP1__E[start + p, 0] = allLegsX[p]
            stackVInd_GP1__E[start + p, 1] = allLegsY[p]
            stackVInd_GP1__E[start + p, 2] = allLegsZ[p]
            # Changed for speed: unpack this point's two vertex hit counts from
            # packed_vertex_hits = start_count + end_count * _VERTEX_HIT_BASE. The float
            # holds a whole number exactly, so converting it to int64 loses nothing. The
            # remainder after dividing by the base is the start count (always below the
            # base, see _VERTEX_HIT_BASE), and the quotient is the end count.
            # octowrap: off
            # point_singularity_counts[start + p, 1] = start_vertex_counts[p]
            # point_singularity_counts[start + p, 2] = end_vertex_counts[p]
            # octowrap: on
            packed_hits = np.int64(packed_vertex_hits[p])
            point_singularity_counts[start + p, 1] = packed_hits % _VERTEX_HIT_BASE
            point_singularity_counts[start + p, 2] = packed_hits // _VERTEX_HIT_BASE

    # Aggregate per point singularity counts into the output array.
    for k in range(1, 3):
        for p in range(num_points):
            singularity_counts[k] += point_singularity_counts[p, k]

    return stackVInd_GP1__E


@njit(cache=True, fastmath=False, parallel=True)
def _expanded_velocities_from_line_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of line vortices and
    finds the induced velocity at every point due to each line vortex.

    This function uses a modified version of the Biot-Savart law to create a smooth
    induced velocity decay based on a line vortex's core radius. The core radius grows
    from an initial value based on the line vortex's age.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    **Citation:**

    Core radius equation adapted from Eq. 3 of: "A Reynolds Number-Based Blade Tip
    Vortex Model"

    Authors: Manikandan Ramasamy and J. Gordon Leishman

    Biot-Savart equation adapted from: "Extended Unsteady Vortex-Lattice Method for
    Insect Flapping Wings"

    Authors: Anh Tuan Nguyen, Joong-Kwan Kim, Jong-Seob Han, and Jae-Hung Han

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackSlvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of M
        line vortices' starting vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackElvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M line vortices' ending vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M line
        vortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        line vortices. Pass zeros for bound line vortices, which then take the kernels'
        numerical floor. For wake line vortices, a reasonable value based on results
        from Ramasamy and Leishman (2007) is 3.0% the chord length of the parent Wing.
        The units are in meters.
    :param singularity_counts: A (3,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] point
        on start vertex, [2] point on end vertex. Counts are incremented in place and
        accumulate across calls.
    :param ages: For bound line vortices, this must be None. For line vortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M line vortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M line vortices. The units are in meters per second.
    """
    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Initialize an empty array, which we will fill with the induced velocities (in the
    # first Airplane's geometry axes, observed from the Earth frame).
    gridVInd_GP1__E = np.zeros((num_points, num_vortices, 3))

    # If the user didn't specify any ages, set the age of each line vortex to 0.0
    # seconds.
    if ages is None:
        ages = np.zeros(num_vortices)

    # Pre compute per line vortex quantities in a serial pass. This hoists invariant
    # work out of the parallel point loop and correctly counts degenerate filaments,
    # which are a per line vortex property, not per point.
    vortex_valid = np.empty(num_vortices, dtype=np.bool)
    vortex_c1 = np.empty(num_vortices)
    vortex_c2 = np.empty(num_vortices)

    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        # The r0_GP1 vector goes from the line vortex's start point to its end point (in
        # the first Airplane's geometry axes).
        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        # Find r0_GP1's length.
        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        # Skip degenerate filaments where the start and end points coincide.
        if r0 < _EPS:
            singularity_counts[0] += 1
            vortex_valid[vortex_id] = False
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        age = ages[vortex_id]

        # Clamp the initial core radius to the numerical floor.
        r_c0 = max(r_c0s[vortex_id], _CORE_FRACTION * r0)

        # Calculate the radius of the line vortex's core squared.
        r_c_sq = r_c0**2.0 + _FOUR_LAMB * (nu + _SQUIRE * abs(strength)) * age

        vortex_c1[vortex_id] = strength / _FOUR_PI
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Use per point singularity counts to avoid write races in the parallel loop. Index
    # mapping: [1] point on start vertex, [2] point on end vertex. Index [0] (degenerate
    # filament) is handled in the serial pre pass above.
    point_singularity_counts = np.zeros((num_points, 3), dtype=np.int64)

    # Numba does not annotate prange.__new__, so mypy cannot tell that calling prange
    # returns an iterable. Once this is fixed upstream, warn_unused_ignores will flag
    # the ignore below as unused.
    for point_id in prange(num_points):  # type: ignore[attr-defined]
        P_GP1_CgP1 = stackP_GP1_CgP1[point_id]

        for vortex_id in range(num_vortices):
            if not vortex_valid[vortex_id]:
                continue

            Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
            Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

            # The r0_GP1 vector goes from the line vortex's start point to its end point
            # (in the first Airplane's geometry axes).
            r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
            r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
            r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

            # The r1_GP1 vector goes from P_GP1_CgP1 to the line vortex's start point
            # (in the first Airplane's geometry axes).
            r1X_GP1 = Slvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r1Y_GP1 = Slvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r1Z_GP1 = Slvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r2_GP1 vector goes from P_GP1_CgP1 to the line vortex's end point (in
            # the first Airplane's geometry axes).
            r2X_GP1 = Elvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r2Y_GP1 = Elvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r2Z_GP1 = Elvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r3_GP1 vector is the cross product of r1_GP1 and r2_GP1 (in the first
            # Airplane's geometry axes). It is computed as r1_GP1 x r0_GP1, which is
            # equal because r2_GP1 = r1_GP1 + r0_GP1, but avoids cancellation when the
            # point is far from the line vortex relative to its length.
            r3X_GP1 = r1Y_GP1 * r0Z_GP1 - r1Z_GP1 * r0Y_GP1
            r3Y_GP1 = r1Z_GP1 * r0X_GP1 - r1X_GP1 * r0Z_GP1
            r3Z_GP1 = r1X_GP1 * r0Y_GP1 - r1Y_GP1 * r0X_GP1

            # Find the lengths of r1_GP1 and r2_GP1.
            r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
            r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)

            # Cache squared length of r3_GP1 as it is used in the c_4 calculation.
            r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0

            # Cache r1 * r2 as it is used twice in the c_4 calculation.
            r1_times_r2 = r1 * r2

            c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

            # The denominator is zero only if the point is on a vertex, as the core
            # radius is always positive. Skip these true singularities.
            denominator = r1_times_r2 * (r3_sq + vortex_c2[vortex_id])
            if denominator == 0.0:
                if r1 == 0.0:
                    point_singularity_counts[point_id, 1] += 1
                elif r2 == 0.0:
                    point_singularity_counts[point_id, 2] += 1
                continue

            # Find r1 * r2 - c_3 in the form that doesn't cancel for this angle. For c_3
            # >= 0.0 (theta <= 90 degrees), use the identity r1 * r2 - c_3 = r3_sq / (r1
            # * r2 + c_3). For c_3 < 0.0, the direct difference is safe.
            if c_3 >= 0.0:
                numerator = r3_sq / (r1_times_r2 + c_3)
            else:
                numerator = r1_times_r2 - c_3

            c_4 = vortex_c1[vortex_id] * (r1 + r2) * numerator / denominator
            gridVInd_GP1__E[point_id, vortex_id, 0] = c_4 * r3X_GP1
            gridVInd_GP1__E[point_id, vortex_id, 1] = c_4 * r3Y_GP1
            gridVInd_GP1__E[point_id, vortex_id, 2] = c_4 * r3Z_GP1

    # Aggregate per point singularity counts into the output array.
    for k in range(1, 3):
        for p in range(num_points):
            singularity_counts[k] += point_singularity_counts[p, k]

    return gridVInd_GP1__E
