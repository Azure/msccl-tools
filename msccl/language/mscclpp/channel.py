# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass
from enum import Enum

from msccl.language.buffer import Buffer


class ChannelType(Enum):
    proxy = "proxy"
    sm = "sm"
    none = "none"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Channel:
    srcBuffer: Buffer
    dstBuffer: Buffer
    type: ChannelType
    connected_to: int
