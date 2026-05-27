"""Memory profile from a MemTracker run — per-microbatch footprint, scaled by in-flight count."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MemoryProfile:
    """Per-microbatch memory footprint for one PP stage.

    Persistent categories (Parameter, Buffer, Gradient, Optstate, Other) are
    resident once. Activation scales with the number of in-flight microbatches;
    Temp (one backward at a time) is added once. The peak for a stage is:

        peak = sum(persistent) + in_flight * act_bytes_per_mb + temp_bytes
    """
    persistent_by_type: dict[str, int] = field(default_factory=dict)
    act_bytes_per_mb: int = 0
    temp_bytes: int = 0
    peak_module: str = ""

    def peak_bytes(self, in_flight: int) -> int:
        return (sum(self.persistent_by_type.values())
                + in_flight * self.act_bytes_per_mb
                + self.temp_bytes)

    def peak_gb(self, in_flight: int) -> float:
        return self.peak_bytes(in_flight) / 1e9

    def peak_by_type(self, in_flight: int) -> dict[str, int]:
        d = dict(self.persistent_by_type)
        d["Activation"] = in_flight * self.act_bytes_per_mb
        d["Temp"] = self.temp_bytes
        return d

    def to_dict(self) -> dict:
        return {
            "persistent_by_type": dict(self.persistent_by_type),
            "act_bytes_per_mb": self.act_bytes_per_mb,
            "temp_bytes": self.temp_bytes,
            "peak_module": self.peak_module,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryProfile":
        return cls(
            persistent_by_type={k: int(v) for k, v in d.get("persistent_by_type", {}).items()},
            act_bytes_per_mb=int(d.get("act_bytes_per_mb", 0)),
            temp_bytes=int(d.get("temp_bytes", 0)),
            peak_module=d.get("peak_module", ""),
        )

    @classmethod
    def from_mem_tracker(cls, mt) -> "MemoryProfile":
        """Build a MemoryProfile from a MemTracker after a num_microbatches=1 pass.

        Buckets categories into persistent (Parameter, Buffer, Gradient,
        Optstate, Other) vs per-microbatch (Activation -> act_bytes_per_mb,
        Temp -> temp_bytes). Sums across devices. peak_module = module with the
        highest local_peak.
        """
        from .mem_tracker import _MemRefType
        persistent_names = {"Parameter", "Buffer", "Gradient", "Optstate", "Other"}
        peak_snap = mt.get_tracker_snapshot("peak")
        persistent: dict[str, int] = {}
        act = 0
        temp = 0
        for dev_snap in peak_snap.values():
            for key, val in dev_snap.items():
                if key == "Total":
                    continue
                name = key.value if isinstance(key, _MemRefType) else str(key)
                if name == "Activation":
                    act += int(val)
                elif name == "Temp":
                    temp += int(val)
                elif name in persistent_names:
                    persistent[name] = persistent.get(name, 0) + int(val)
        peak_module = ""
        best = -1
        for mod_stats in mt.memory_tracking.values():
            local = max(mod_stats.local_peak.values(), default=0)
            if local > best:
                best = local
                peak_module = mod_stats.mod_fqn
        return cls(
            persistent_by_type=persistent,
            act_bytes_per_mb=act,
            temp_bytes=temp,
            peak_module=peak_module,
        )
