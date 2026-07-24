"""
Match library thread pools to the container's actual CPU allowance.

OpenCV, ONNX Runtime, OpenBLAS and OpenMP all size their pools from the
*host's* core count. Inside a container that host may have 64 cores while the
cgroup allows 6 — so each library happily spawns dozens of threads that then
fight over a handful of cores. The result is heavy oversubscription: the
process shows >1000% CPU, the cgroup throttles it, and every stage of the
pipeline slows down together, including code that was never touched.

This must run **before** numpy, cv2 or onnxruntime are imported: those
libraries read their thread-count environment variables once, at import.
"""

from __future__ import annotations

import os


def detect_cpu_quota() -> int:
    """Cores actually available to this container.

    Reads the cgroup quota rather than trusting os.cpu_count(), which reports
    the host. Some hosts (Runpod among them) don't expose a quota at all even
    though the container is limited, so CPU_LIMIT can be set explicitly — and
    should be, to whatever vCPU count the pod was provisioned with.
    """
    explicit = os.environ.get("CPU_LIMIT")
    if explicit:
        try:
            return max(1, int(explicit))
        except ValueError:
            pass
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except Exception:
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, quota // period)
    except Exception:
        pass

    # Respects taskset/affinity where cgroups aren't readable.
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def apply(reserve: int = 1) -> int:
    """Cap every thread pool. Returns the limit applied.

    One core is held back for the asyncio loop, LiveKit's own threads and the
    audio pipeline, so inference can't starve the parts that move frames in
    and out.
    """
    quota = detect_cpu_quota()
    limit = max(1, quota - reserve)

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(limit))

    # OpenCV reads this at import; setNumThreads is applied again afterwards.
    os.environ.setdefault("OPENCV_FOR_THREADS_NUM", str(limit))

    print(
        f"CPU limits: cgroup allows {quota} core(s); thread pools capped at "
        f"{limit} (1 reserved for I/O)",
        flush=True,
    )
    return limit


def apply_runtime_limits(limit: int) -> None:
    """Called after imports for libraries with a runtime setter."""
    try:
        import cv2

        cv2.setNumThreads(limit)
    except Exception:
        pass
