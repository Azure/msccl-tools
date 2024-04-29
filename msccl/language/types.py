# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from enum import Enum

from msccl.language.buffer import Buffer


@dataclass
class Program:
    name: str
    collective: str
    inplace: bool
    protocol: str
    gpus: list = field(default_factory=list)
    num_chunk_groups: int = 1


@dataclass
class Gpu:
    rank: int
    threadblocks: list = field(default_factory=list)

    # From ncclize
    precopies: list = field(default_factory=list)
    postcopies: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    input_chunks: int = 0
    output_chunks: int = 0
    scratch_chunks: int = 0
    scratch: dict = field(default_factory=dict)

    def scratch_size(self):
        return max((idx for addr, idx in self.scratch.items()), default=-1) + 1


@dataclass
class Threadblock:
    id: int = -1
    channel: int = -1
    send: int = -1
    recv: int = -1
    ops: list = field(default_factory=list)
    rbid: int = -1  # threadblock id of the receiver

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class ChunkInstruction(Enum):
    start = "start"
    reduce = "reduce"
    send = "send"

    def __str__(self):
        return self.value


class ThreadblockPolicy(Enum):
    auto = "auto"
    manual = "manual"

    def __str__(self):
        return self.value


class ReplicationPolicy(Enum):
    # this means each instance deal with the different chunk
    # Chunk A, Chunk B -> Chunk A0, Chunk B0, Chunk A1, Chunk B1
    duplicated = "duplicated"
    # this means each instance deal with the different chunk in interleaved way
    # Chunk A, Chunk B -> Chunk A0, Chunk A1, Chunk B0, Chunk B1
    interleaved = "interleaved"

    def __str__(self):
        return self.value


class Instruction(Enum):
    delete = "d"
    start = "st"
    nop = "nop"
    send = "s"
    recv = "r"
    recv_copy_send = "rcs"
    recv_reduce_send = "rrs"
    recv_reduce_copy = "rrc"
    recv_reduce_copy_send = "rrcs"
    copy = "cpy"
    reduce = "re"

    def __str__(self):
        return self.value


@dataclass
class ChunkRef:
    rank: int
    buffer: Buffer
    index: int
    size: int

    def __hash__(self):
        return hash((self.rank, self.buffer, self.index, self.size))


@dataclass
class Op:
    inst: Instruction
    rank: int
    src: ChunkRef
    dst: ChunkRef
    depends: list = field(default_factory=list)
    step: int = -1  # Step in the TB
    tb: int = -1  # TB this op is assigned to
    prev: list = field(default_factory=list)  # List of instructions that happen before
    next: list = field(default_factory=list)  # List of instructions that happen after
    num: int = -1
    chunk_step: int = -1
    priority: int = -1
    recv_match = None
    send_match = None
    channel: int = -1
    srcs: list = field(default_factory=list)
    dsts: list = field(default_factory=list)

    def cnt(self):
        if self.src:
            if self.dst:
                assert self.src.size == self.dst.size
            return self.src.size
        elif self.dst:
            return self.dst.size
        else:
            return 0

    def is_send(self):
        return (
            self.inst == Instruction.send
            or self.inst == Instruction.recv_reduce_copy_send
            or self.inst == Instruction.recv_copy_send
            or self.inst == Instruction.recv_reduce_send
        )

    def is_recv(self):
        return (
            self.inst == Instruction.recv
            or self.inst == Instruction.recv_reduce_copy
            or self.inst == Instruction.recv_reduce_copy_send
            or self.inst == Instruction.recv_copy_send
            or self.inst == Instruction.recv_reduce_send
        )

    def is_fused(self):
        return (
            self.inst == Instruction.recv_reduce_copy_send
            or self.inst == Instruction.recv_copy_send
            or self.inst == Instruction.recv_reduce_send
        )

    def is_local(self):
        return self.inst == Instruction.copy or self.inst == Instruction.reduce

    def peer(self):
        if self.inst == Instruction.send:
            return self.dst.rank
        elif self.inst == Instruction.recv:
            return self.src.rank
        else:
            return None

    def send_peer(self):
        if self.is_send():
            return self.dst.rank
        return -1

    def recv_peer(self):
        if self.is_recv():
            return self.src.rank
        return -1

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        # Ordering of operations
        # 1. Lower chunk step 2. Higher priority 3. Lower src index
        if self.chunk_step == other.chunk_step:
            if self.priority == other.priority:
                return self.src.index < other.src.index
            return self.priority > other.priority
        return self.chunk_step < other.chunk_step

    def __gt__(self, other):
        return not self < other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Op({self.inst}, {self.rank}, {self.src}, {self.dst}, step:{self.step}, tb:{self.tb})"
