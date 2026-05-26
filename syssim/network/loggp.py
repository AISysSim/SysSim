"""LogGP performance model parameters for point-to-point communication.

The LogGP model characterizes message-passing performance with parameters:
- L (latency): Network latency for sending a minimal message (seconds)
- o (overhead): CPU overhead per send/receive operation (seconds)
- g (gap): Minimum time between consecutive sends (seconds) - Hoefler model
- G = 1/bandwidth (gap per byte in seconds/byte)
- P (processors): Number of processors (not stored here)

Two model variants:
1. **Simplified 3-parameter model** (g=0): T = L + 2*o + (m-1)*G
   - Used when g is negligible or absorbed into other parameters
   - Backward compatible default

2. **Hoefler 4-parameter model** (g>0): T = L + 2*o + g + (m-1)*G
   - Separates base gap (g) from per-byte gap (G)
   - Measured via PRTT (Parametrized Round Trip Time) method
   - More accurate for protocol changes (eager vs rendezvous)

The factor of 2 on overhead accounts for both send and receive overhead.
The (m-1) factor comes from pipelining: first byte incurs L+2o, remaining bytes incur G.

References:
- "LogGP: Incorporating Long Messages into the LogP model" (Alexandrov et al., 1995)
- "LogGP in Theory and Practice" (Hoefler et al., 2009)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogGPParams:
    """LogGP performance model parameters.

    Attributes:
        L: Latency for sending a minimal message (seconds)
        o: CPU overhead per send/receive operation (seconds)
        G: Gap per byte, equal to 1/bandwidth (seconds/byte)
        g: Base gap for consecutive sends (seconds), default 0.0 for simplified model

    Example (simplified 3-parameter model):
        >>> # 25 GB/s NVLink with 1μs latency, 5μs overhead
        >>> loggp = LogGPParams(L=1e-6, o=5e-6, G=1/(25e9))
        >>> loggp.alpha  # Fixed pipeline overhead
        1.1e-05
        >>>
        >>> # Time to send 1 MB message
        >>> msg_size = 1e6
        >>> time = loggp.message_time(msg_size)
        >>> print(f"{time*1e3:.3f} ms")
        0.051 ms

    Example (Hoefler 4-parameter model):
        >>> # Profiled parameters with explicit gap
        >>> loggp = LogGPParams(L=1.5e-6, o=7e-6, G=4e-11, g=2e-6)
        >>> loggp.alpha  # L + 2*o + g
        1.7e-05
    """

    L: float  # latency (seconds)
    o: float  # per-message CPU overhead (seconds)
    G: float  # gap per byte (seconds/byte)
    g: float = 0.0  # base gap (seconds), default 0 for backward compatibility

    @property
    def alpha(self) -> float:
        """Fixed pipeline overhead (L + 2*o + g) in seconds.

        This is the overhead incurred once per message, independent of size.
        The remaining time scales linearly with message size via the G parameter.

        For Hoefler 4-parameter model: includes base gap g
        For simplified 3-parameter model: g=0, reduces to L + 2*o

        Returns:
            L + 2*o + g in seconds
        """
        return self.L + 2 * self.o + self.g

    def message_time(self, size_bytes: float) -> float:
        """Compute time to send a message of given size.

        Formula (Hoefler 4-parameter): T = L + 2*o + g + (size_bytes - 1) * G
        Formula (simplified 3-parameter): T = L + 2*o + (size_bytes - 1) * G (when g=0)

        Args:
            size_bytes: Message size in bytes

        Returns:
            Transfer time in seconds

        Example (simplified model):
            >>> loggp = LogGPParams(L=1e-6, o=5e-6, G=1/(25e9))
            >>> loggp.message_time(1e6)  # 1 MB
            5.0999e-05

        Example (Hoefler model):
            >>> loggp = LogGPParams(L=1.5e-6, o=7e-6, G=4e-11, g=2e-6)
            >>> loggp.message_time(1e6)  # 1 MB
            5.699e-05
        """
        if size_bytes <= 0:
            return 0.0
        return self.alpha + (size_bytes - 1) * self.G
