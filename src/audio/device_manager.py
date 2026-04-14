"""
Audio Device Manager

Handles audio device enumeration and selection using sounddevice/PortAudio.
Works with PipeWire, PulseAudio, and ALSA on Linux.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import sounddevice as sd


@dataclass
class AudioDevice:
    """Represents an audio device."""
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_input(self) -> bool:
        """Check if device supports input."""
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """Check if device supports output."""
        return self.max_output_channels > 0

    def __str__(self) -> str:
        direction = []
        if self.is_input:
            direction.append("IN")
        if self.is_output:
            direction.append("OUT")
        markers = []
        if self.is_default_input:
            markers.append("*default input")
        if self.is_default_output:
            markers.append("*default output")
        marker_str = f" ({', '.join(markers)})" if markers else ""
        return f"[{self.index}] {self.name} [{'/'.join(direction)}]{marker_str}"


class AudioDeviceManager:
    """
    Manages audio device discovery and selection.

    This class provides methods to enumerate available audio devices,
    identify default devices, and validate device configurations.
    """

    def __init__(self):
        """Initialize the device manager."""
        self._devices: List[AudioDevice] = []
        self._host_apis: Dict[int, str] = {}
        self.refresh()

    def refresh(self) -> None:
        """Refresh the list of available audio devices."""
        self._devices = []
        self._host_apis = {}

        # Get host API info
        for i in range(sd.query_hostapis().__len__()):
            api_info = sd.query_hostapis(i)
            self._host_apis[i] = api_info['name']

        # Get default device indices
        try:
            default_input, default_output = sd.default.device
        except Exception:
            default_input, default_output = None, None

        # Enumerate all devices
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            host_api_name = self._host_apis.get(dev['hostapi'], 'Unknown')

            device = AudioDevice(
                index=i,
                name=dev['name'],
                host_api=host_api_name,
                max_input_channels=dev['max_input_channels'],
                max_output_channels=dev['max_output_channels'],
                default_sample_rate=dev['default_samplerate'],
                is_default_input=(i == default_input),
                is_default_output=(i == default_output),
            )
            self._devices.append(device)

    @property
    def devices(self) -> List[AudioDevice]:
        """Get all available devices."""
        return self._devices

    def get_input_devices(self) -> List[AudioDevice]:
        """Get all devices that support audio input."""
        return [d for d in self._devices if d.is_input]

    def get_output_devices(self) -> List[AudioDevice]:
        """Get all devices that support audio output."""
        return [d for d in self._devices if d.is_output]

    def get_default_input(self) -> Optional[AudioDevice]:
        """Get the default input device."""
        for d in self._devices:
            if d.is_default_input:
                return d
        # Fallback: return first input device
        inputs = self.get_input_devices()
        return inputs[0] if inputs else None

    def get_default_output(self) -> Optional[AudioDevice]:
        """Get the default output device."""
        for d in self._devices:
            if d.is_default_output:
                return d
        # Fallback: return first output device
        outputs = self.get_output_devices()
        return outputs[0] if outputs else None

    def get_device_by_name(self, name: str) -> Optional[AudioDevice]:
        """
        Find a device by name (partial match).

        Args:
            name: Device name or substring to match

        Returns:
            AudioDevice if found, None otherwise
        """
        name_lower = name.lower()

        # Exact match first
        for d in self._devices:
            if d.name.lower() == name_lower:
                return d

        # Partial match
        for d in self._devices:
            if name_lower in d.name.lower():
                return d

        return None

    def get_device_by_index(self, index: int) -> Optional[AudioDevice]:
        """
        Get a device by its index.

        Args:
            index: Device index

        Returns:
            AudioDevice if found, None otherwise
        """
        for d in self._devices:
            if d.index == index:
                return d
        return None

    def resolve_device(
        self,
        device_spec: str,
        direction: str = 'input'
    ) -> Optional[AudioDevice]:
        """
        Resolve a device specification to an AudioDevice.

        Args:
            device_spec: Device name, index (as string), or 'default'
            direction: 'input' or 'output'

        Returns:
            AudioDevice if found, None otherwise
        """
        if device_spec == 'default':
            if direction == 'input':
                return self.get_default_input()
            else:
                return self.get_default_output()

        # Try as index
        try:
            index = int(device_spec)
            device = self.get_device_by_index(index)
            if device:
                return device
        except ValueError:
            pass

        # Try as name
        return self.get_device_by_name(device_spec)

    def validate_input_device(
        self,
        device: AudioDevice,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> bool:
        """
        Validate that a device can be used for input with given parameters.

        Args:
            device: Device to validate
            sample_rate: Required sample rate
            channels: Required number of channels

        Returns:
            True if device supports the configuration
        """
        if not device.is_input:
            return False

        if device.max_input_channels < channels:
            return False

        # Test if sample rate is supported
        try:
            sd.check_input_settings(
                device=device.index,
                samplerate=sample_rate,
                channels=channels,
            )
            return True
        except Exception:
            return False

    def validate_output_device(
        self,
        device: AudioDevice,
        sample_rate: int = 22050,
        channels: int = 1
    ) -> bool:
        """
        Validate that a device can be used for output with given parameters.

        Args:
            device: Device to validate
            sample_rate: Required sample rate
            channels: Required number of channels

        Returns:
            True if device supports the configuration
        """
        if not device.is_output:
            return False

        if device.max_output_channels < channels:
            return False

        try:
            sd.check_output_settings(
                device=device.index,
                samplerate=sample_rate,
                channels=channels,
            )
            return True
        except Exception:
            return False

    def print_devices(self, filter_type: str = 'all') -> None:
        """
        Print a formatted list of devices.

        Args:
            filter_type: 'all', 'input', or 'output'
        """
        if filter_type == 'input':
            devices = self.get_input_devices()
            print("Input Devices:")
        elif filter_type == 'output':
            devices = self.get_output_devices()
            print("Output Devices:")
        else:
            devices = self._devices
            print("All Audio Devices:")

        print("-" * 60)
        for d in devices:
            print(f"  {d}")
        print("-" * 60)


# Convenience functions for module-level access
_manager: Optional[AudioDeviceManager] = None


def _get_manager() -> AudioDeviceManager:
    """Get or create the global device manager."""
    global _manager
    if _manager is None:
        _manager = AudioDeviceManager()
    return _manager


def list_input_devices() -> List[AudioDevice]:
    """List all available input devices."""
    return _get_manager().get_input_devices()


def list_output_devices() -> List[AudioDevice]:
    """List all available output devices."""
    return _get_manager().get_output_devices()


def get_default_input_device() -> Optional[AudioDevice]:
    """Get the default input device."""
    return _get_manager().get_default_input()


def get_default_output_device() -> Optional[AudioDevice]:
    """Get the default output device."""
    return _get_manager().get_default_output()
