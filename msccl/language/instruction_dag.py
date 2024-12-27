# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from collections import defaultdict

from msccl.language.buffer import Buffer
from msccl.language.types import ChunkRef, Gpu, Instruction, Op, ReplicationPolicy, Threadblock


def remove_op(op: Op):
    for p in op.prev:
        p.next.remove(op)
        p.next += op.next
        p.next = list(set(p.next))

    for n in op.next:
        n.prev.remove(op)
        n.prev = op.prev.union(n.prev)

    op.next = []
    op.prev = []


def merge_op(op: Op, other_op: Op):
    if other_op in op.next:
        op.next.remove(other_op)
        other_op.prev.remove(op)
    for p in other_op.prev:
        p.next.remove(other_op)
        p.next.append(op)

    for n in other_op.next:
        n.prev.remove(other_op)
        n.prev.add(op)

    op.prev = op.prev.union(other_op.prev)
    op.next = list(set(op.next + other_op.next))


def circular_dep_after_merge(op: Op, other_op: Op):
    root = set([op, other_op])
    frontier = set(op.next)
    if other_op in frontier:
        frontier.remove(other_op)
    frontier = list(frontier.union(other_op.next))
    while len(frontier) > 0:
        current = frontier[0]
        for n in current.next:
            # The root node will be visited again if there is a circular dependency
            if n in root:
                return True
            frontier.append(n)
        frontier = frontier[1:]

"""
For case: op2.prev = [op1, op3]. op1.next = [op2]. op3.next = [op2]. And op1 and op2 are satisfied to merge.
We only apply the merge if all previous ops of op2 are visited. (op1 is the last previous op of op2).
"""
def all_prevs_visited_after_merge(op: Op, other_op: Op):
    step = op.step
    for prev in other_op.prev:
        if prev.step > step:
            return False
    return True

def same_tb(op1: Op, op2: Op):
    return op1.tb == op2.tb and op1.channel == op2.channel


def same_count(op1: Op, op2: Op):
    return op1.cnt() == op2.cnt()


def same_buf_dst(op1: Op, op2: Op):
    return op1.dst.buffer == op2.dst.buffer and op1.dst.index == op2.dst.index


def same_src_dst_buffer_type(op1: Op, op2: Op):
    return op1.src.buffer == op2.src.buffer and op1.dst.buffer == op2.dst.buffer


def buf_dst_src_match(op1: Op, op2: Op):
    return op1.dst.buffer == op2.src.buffer and op1.dst.index == op2.src.index


def same_buf_src(op1: Op, op2: Op):
    return op1.src.buffer == op2.src.buffer and op1.src.index == op2.src.index


def same_chan_type(op1: Op, op2: Op):
    return op1.channel_type == op2.channel_type

def same_tb(op1: Op, op2: Op):
    return op1.tb == op2.tb


class InstructionDAG(ABC):
    def __init__(self, num_ranks, buffers):
        self.num_ranks = num_ranks
        self.buffers = buffers
        # State for the actual instruction DAG
        self.operations = {}  # slot -> operations
        self.last_writer = {}  # slot -> last writing op
        self.last_readers = defaultdict(list)  # slot -> list of last reading ops
        # State for the MSCCL-IR
        self.tbs = []
        for _ in range(num_ranks):
            self.tbs.append({})
        self.tb_mapping = {}
        self.num_channels = [1] * num_ranks
        self.tb_steps = [{} for _ in range(num_ranks)]

    # InstructionDAG helper - identifies the dependencies for a write-type operation (recv, copy, rrc, reduce)
    def _write(self, rank, buffer, index, size, op, read=False):
        prev_ops = set()
        for i in range(index, index + size):
            slot = (rank, buffer, i)
            if read:
                assert slot in self.last_writer, f"Destination slot has never been written before a reduce {op}"

            # First write to this slot
            if slot not in self.operations:
                self.operations[slot] = op

            # If there are active readers - these are the previous operations
            # Else the previous operation is the last write (if there is one)
            readers = self.last_readers[slot]
            if len(readers) > 0:
                prev_ops.update(readers)
            elif slot in self.last_writer:
                prev_ops.add(self.last_writer[slot])

            # Set the last_writer to this op, and clear all readers
            self.last_writer[slot] = op
            self.last_readers[slot] = []

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG helper - identifies the dependencies for read-type operations (send, copy, reduce)
    def _read(self, rank, buffer, index, size, op):
        prev_ops = set()
        for i in range(index, index + size):
            slot = (rank, buffer, i)
            assert slot in self.last_writer, f"Slot has never been written before a read-type {op}"
            # The previous operation for a reader is the last write to the slot
            writer = self.last_writer[slot]
            prev_ops.add(writer)
            self.last_readers[slot].append(op)

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    def _infer_dependencies(self):
        visited = set()
        for _, op in self.operations.items():
            if op in visited:
                continue
            frontier = [op]
            while len(frontier) > 0:
                op = frontier[0]
                if op in visited:
                    frontier = frontier[1:]
                    continue
                # Dependencies for every op is the same as the ops that are stored in prev
                # Filter out dependencies that are satisified by tbs executing ops sequentially
                # If multiple dependent ops from the same tb keep the one that happens last
                depends = {}
                for dep_op in list(op.prev):
                    if dep_op.inst != Instruction.start:
                        tb = dep_op.tb
                        if tb not in depends or dep_op.step > depends[tb].step:
                            depends[tb] = dep_op
                op.depends = list(depends.values())
                visited.add(op)
                frontier = frontier[1:] + op.next

    # Convert local scratch buffers to index into one global scratch buffer
    def _lower_chunk(self, chunk):
        if chunk is not None and chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            buffer = self.buffers[chunk.rank][chunk.buffer].get_buffer()
            index = self.buffers[chunk.rank][chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(chunk.rank, buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def _lower_buffers(self, instances):
        for rank_buffers in self.buffers:
            offset = 0
            for key, buf in rank_buffers.items():
                if key is not Buffer.input and key is not Buffer.output:
                    buf.set_offset(offset)
                    offset += buf.instance_size() * instances

    # Preprocess the threadblocks for lowering into xml
    def _lower_tbs(self):
        gpus = []
        for rank, rank_tbs in enumerate(self.instanced_tbs):
            lowered_tbs = {}
            for tbid, tb in rank_tbs.items():
                for op in tb.ops:
                    op.src = self._lower_chunk(op.src)
                    op.dst = self._lower_chunk(op.dst)
                    srcs = sorted(op.srcs, key=lambda x: x[1])
                    dsts = sorted(op.dsts, key=lambda x: x[1])
                    op.srcs = [self._lower_chunk(src[0]) for src in srcs]
                    op.dsts = [self._lower_chunk(dst[0]) for dst in dsts]
                lowered_tbs[tbid] = tb
            gpus.append(Gpu(rank, list(lowered_tbs.values())))
        return gpus

    # InstructionDAG - builds the roots of the DAG
    def add_start(self, rank, buffer, index, ref):
        slot = (rank, buffer, index)
        op = Op(Instruction.start, rank, ref, ref, next=set(), prev=set(), chunk_step=-1)
        self.operations[slot] = op
        self.last_writer[slot] = op

    def convert_set_list(self):
        ops = []
        visited = set()
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                op.next = list(op.next)
                for o in op.next:
                    ops.append(o)
            elif op.inst != Instruction.copy:
                ops.append(op)

            while len(ops) > 0:
                op = ops[0]
                if op not in visited:
                    visited.add(op)
                    op.next = list(op.next)
                    ops = ops[1:] + op.next
                else:
                    ops = ops[1:]
        return visited

    def lower_pt1(self, instances: int):
        self._infer_dependencies()
        self._lower_buffers(instances)

    def lower_pt2(self, instances: int, replication_policy: ReplicationPolicy):
        self.replicate(instances, replication_policy)
        return self._lower_tbs()

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def replicate(self, instances: int, replication_policy: ReplicationPolicy):
        pass