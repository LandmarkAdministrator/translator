"""Audio management modules for the Church Audio Translator."""

from .device_manager import (
    AudioDeviceManager,
    list_input_devices,
    list_output_devices,
    get_default_input_device,
    get_default_output_device,
)
from .input_stream import AudioInputStream
from .output_stream import AudioOutputStream, SharedStereoOutput, ChannelOutputProxy

__all__ = [
    'AudioDeviceManager',
    'list_input_devices',
    'list_output_devices',
    'get_default_input_device',
    'get_default_output_device',
    'AudioInputStream',
    'AudioOutputStream',
    'SharedStereoOutput',
    'ChannelOutputProxy',
]
