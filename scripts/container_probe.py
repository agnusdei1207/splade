import json
from pathlib import Path


def read_limit(name: str) -> int:
    value = Path("/sys/fs/cgroup", name).read_text(encoding="utf-8").strip()
    return -1 if value == "max" else int(value)


cpu_quota, cpu_period = (
    int(value)
    for value in Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
)
mount_lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
corpus_read_only = any(
    fields[4] == "/corpus" and "ro" in fields[5].split(",")
    for line in mount_lines
    if len(fields := line.split()) > 5
)

print(
    json.dumps(
        {
            "memory_max": read_limit("memory.max"),
            "swap_max": read_limit("memory.swap.max"),
            "cpu_quota": cpu_quota,
            "cpu_period": cpu_period,
            "pids_max": read_limit("pids.max"),
            "corpus_read_only": corpus_read_only,
        }
    )
)
