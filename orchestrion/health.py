"""Shared health states for robot peripherals and monitoring integrations."""

from enum import Enum


class DeviceHealth(str, Enum):
    """Normalized device connectivity and readiness states."""

    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
