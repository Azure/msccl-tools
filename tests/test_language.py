# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import msccl
from msccl.language.types import MscclppInstruction
from msccl.topologies import line, fully_connected
from msccl.language import *
from msccl.language.routines import *
from msccl.language.collectives import *
import os
import pytest


class Send(Collective):
    # Initial state is chunk0 is on rank0 in the input buffer
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            if r == 0:
                for c in range(chunks_per_node):
                    input_buffer[c] = Chunk(r, c, 2, c)
            buffers = {Buffer.input: input_buffer, Buffer.output: output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    # Final state chunk0 from rank0 is in the output buffer of rank2
    def check(self, prog):
        correct = True
        output = prog.buffers[2][Buffer.output]
        for c in range(self.chunk_factor):
            chunk = output[c]
            # Check that we got chunk 0 from rank 0
            if chunk is None or chunk.origin_rank != 0 or chunk.origin_index != 0:
                print(f"Rank 2 chunk {c} is incorrect should be ({0}, {0}) given {chunk}")
                correct = False
        return correct


class Reduce(Collective):
    # Initial state is chunk0,0 is on rank0 in the input buffer
    # and chunk0,1 is on rank1 in the input buffer, etc.
    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None] * chunks_per_node
            output_buffer = [None] * chunks_per_node
            for c in range(chunks_per_node):
                input_buffer[c] = Chunk(r, c, -1, c)
            buffers = {Buffer.input: input_buffer, Buffer.output: output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    # Final state rank2 has a fully reduced chunk from gpus 0, 1, and 2
    def check(self, prog):
        expected_chunk = ReduceChunk(-1, [])
        for r in range(self.num_ranks):
            expected_chunk = expected_chunk.reduce(-1, Chunk(r, 0))

        correct = True
        chunk = prog.buffers[2][Buffer.input][0]
        if chunk is None or chunk != expected_chunk:
            print(f"Rank 2 chunk 0 is incorrect should be ReduceChunk index 0 from all ranks, given {chunk}")
            correct = False
        return correct


def test_send():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Send(num_gpus, chunksperloop, inplace=False)
    with MSCCLProgram("send", topology, collective, instances):
        chunk(0, Buffer.input, 0).copy(1, "scratch").copy(2, Buffer.output, 0)
        assert Check()


def test_reduce():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Reduce(num_gpus, chunksperloop, inplace=True)
    with MSCCLProgram("reduce", topology, collective, instances):
        c10 = chunk(1, Buffer.input, 0).reduce(chunk(0, Buffer.input, 0))
        chunk(2, Buffer.input, 0).reduce(c10)
        assert Check()


def test_local_copy():
    num_gpus = 3
    topology = fully_connected(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Send(num_gpus, chunksperloop, inplace=False)
    with MSCCLProgram("cpy", topology, collective, instances):
        chunk(0, Buffer.input, 0).copy(2, "scratch").copy(2, Buffer.output, 0)
        assert Check()


def test_local_reduce():
    num_gpus = 3
    topology = line(num_gpus)

    chunksperloop = 1
    instances = 1
    collective = Reduce(num_gpus, chunksperloop, inplace=True)
    with MSCCLProgram("local-reduce", topology, collective, instances):
        c = chunk(1, Buffer.input, 0).reduce(chunk(0, Buffer.input, 0))
        c = c.copy(2, "scratch", 0)
        chunk(2, Buffer.input, 0).reduce(c)
        XML()
        assert Check()


def test_scratch_buffers():
    num_gpus = 3
    topology = fully_connected(num_gpus)

    chunksperloop = num_gpus
    instances = 1
    collective = AllReduce(num_gpus, chunksperloop, inplace=False)
    with MSCCLProgram("test", topology, collective, instances):
        chunk(0, Buffer.input, 0).copy(2, "scratch", 2)
        c = chunk(2, "scratch", 2)
        assert c.index == 2
        c = chunk(1, Buffer.input, 0).copy(2, "scratch")
        assert c.index == 3
        XML()


def test_program_order():
    num_gpus = 2
    topology = fully_connected(num_gpus)

    chunksperloop = num_gpus
    instances = 1
    collective = AllReduce(num_gpus, chunksperloop, inplace=False)
    prgm = MSCCLProgram("test", topology, collective, instances)
    with prgm:
        chunk(1, Buffer.input, 0).copy(0, "sc", 1)
        # This send should depend on the send above finishing
        chunk(0, Buffer.input, 0).copy(1, Buffer.input, 0)
    slot = (1, Buffer.input, 0)
    prgm.lower()
    op = prgm.instr_dag.operations[slot]
    assert op.inst == Instruction.start
    assert op.next[0].inst == Instruction.send
    assert op.next[0].next[0].inst == Instruction.recv


def test_allgather():
    topology = fully_connected(2)
    collective = AllGather(2, 1, True)
    with MSCCLProgram("allgather", topology, collective, 1):
        chunk(0, Buffer.input, 0).copy(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).copy(0, Buffer.output, 1)
        assert Check()


def test_reducescatter():
    topology = fully_connected(2)
    collective = ReduceScatter(2, 1, True)
    with MSCCLProgram("reducescatter", topology, collective, 1):
        chunk(1, Buffer.input, 1).reduce(chunk(0, Buffer.input, 1))
        chunk(0, Buffer.input, 0).reduce(chunk(1, Buffer.input, 0))
        assert Check()


def test_alltoall():
    topology = fully_connected(2)
    collective = AllToAll(2, 1, False)
    with MSCCLProgram("alltoall", topology, collective, 1):
        chunk(0, Buffer.input, 0).copy(0, Buffer.output, 0)
        chunk(0, Buffer.input, 1).copy(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).copy(0, Buffer.output, 1)
        chunk(1, Buffer.input, 1).copy(1, Buffer.output, 1)
        assert Check()


def test_allreduce():
    topology = fully_connected(2)
    collective = AllReduce(2, 2, True)
    with MSCCLProgram("allreduce", topology, collective, 1):
        chunk(1, Buffer.output, 0).reduce(chunk(0, Buffer.input, 0)).copy(0, Buffer.input, 0)
        chunk(0, Buffer.input, 1).reduce(chunk(1, Buffer.input, 1)).copy(1, Buffer.input, 1)
        assert Check()


def test_instruction_fusion():
    topology = fully_connected(3)
    collective = AllReduce(3, 3, True)
    prgm = MSCCLProgram("allreduce", topology, collective, 1, threadblock_policy=ThreadblockPolicy.manual)
    with prgm:
        c01 = chunk(1, Buffer.input, 0, 3).reduce(chunk(0, Buffer.input, 0, 3), sendtb=0, recvtb=0, ch=0)
        c012 = chunk(2, Buffer.input, 0, 3).reduce(c01, sendtb=0, recvtb=0, ch=0)
        c012.copy(0, Buffer.input, 0, sendtb=0, recvtb=0, ch=0).copy(1, Buffer.input, 0, sendtb=0, recvtb=0, ch=0)
        assert Check()
    lowered_prgm = prgm.lower()
    assert lowered_prgm.gpus[0].threadblocks[0].ops[0].inst == Instruction.send
    assert lowered_prgm.gpus[0].threadblocks[0].ops[1].inst == Instruction.recv_copy_send
    assert lowered_prgm.gpus[1].threadblocks[0].ops[0].inst == Instruction.recv_reduce_send
    assert lowered_prgm.gpus[1].threadblocks[0].ops[1].inst == Instruction.recv
    assert lowered_prgm.gpus[2].threadblocks[0].ops[0].inst == Instruction.recv_reduce_copy_send


def test_instruction_fusion_mscclpp():
    topology = fully_connected(3)
    collective = AllReduce(3, 3, True)
    prgm = MSCCLPPProgram("allreduce", topology, collective, 1)
    with prgm:
        c01 = chunk(1, Buffer.input, 0, 3).reduce(chunk(0, Buffer.input, 0, 3), recvtb=0)
        c01.signal(2, Buffer.input, 0, sendtb=0)
        c012 = chunk(2, Buffer.input, 0, 3)
        c012.wait(1, Buffer.input, 0, recvtb=0)
        c012.reduce(c01, recvtb=0).put(0, Buffer.input, 0, sendtb=0)
        c012.signal(0, Buffer.input, 0, sendtb=0)
        c0 = chunk(0, Buffer.input, 0, 3)
        c0.wait(2, Buffer.input, 0, recvtb=0)
        c0.put(1, Buffer.input, 0, sendtb=0)
        assert Check()
    lowered_prgm = prgm.lower()
    assert lowered_prgm.gpus[0].threadblocks[0].ops[0].inst == MscclppInstruction.wait
    assert lowered_prgm.gpus[0].threadblocks[0].ops[1].inst == MscclppInstruction.put
    assert lowered_prgm.gpus[1].threadblocks[0].ops[0].inst == MscclppInstruction.read_reduce_copy
    assert lowered_prgm.gpus[1].threadblocks[0].ops[1].inst == MscclppInstruction.signal
    assert lowered_prgm.gpus[2].threadblocks[0].ops[0].inst == MscclppInstruction.wait
    assert lowered_prgm.gpus[2].threadblocks[0].ops[1].inst == MscclppInstruction.read_reduce_copy_send
    assert lowered_prgm.gpus[2].threadblocks[0].ops[2].inst == MscclppInstruction.signal


def test_instruction_fusion_put_signal_flush_mscclpp():
    topology = fully_connected(2)
    collective = SendRecv(2, 1, False)
    prgm = MSCCLPPProgram("allreduce", topology, collective, 1)
    with prgm:
        c0 = chunk(0, Buffer.input, 0)
        c0.put(1, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c0.signal(1, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c0.flush(1, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c1 = chunk(1, Buffer.input, 0)
        c1.put(0, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c1.signal(0, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c1.flush(0, "scratch", 1, sendtb=0, chan_type=ChannelType.proxy)
        c01 = chunk(0, "scratch", 1)
        c01.wait(1, Buffer.input, 0, recvtb=0, chan_type=ChannelType.proxy)
        c01.copy(0, Buffer.output, 0, sendtb=0)
        c10 = chunk(1, "scratch", 1)
        c10.wait(0, Buffer.input, 0, recvtb=0, chan_type=ChannelType.proxy)
        c10.copy(1, Buffer.output, 0, sendtb=0)
        assert Check()
    lowered_prgm = prgm.lower()
    assert lowered_prgm.gpus[0].threadblocks[0].ops[0].inst == MscclppInstruction.put_with_signal_and_flush
    assert lowered_prgm.gpus[0].threadblocks[0].ops[1].inst == MscclppInstruction.wait
    assert lowered_prgm.gpus[0].threadblocks[0].ops[2].inst == MscclppInstruction.copy
    assert lowered_prgm.gpus[1].threadblocks[0].ops[0].inst == MscclppInstruction.put_with_signal_and_flush
    assert lowered_prgm.gpus[1].threadblocks[0].ops[1].inst == MscclppInstruction.wait
    assert lowered_prgm.gpus[1].threadblocks[0].ops[2].inst == MscclppInstruction.copy


def test_instruction_fusion_multi_deps_mscclpp():
    topology = fully_connected(3)
    collective = AllReduce(3, 1, True)
    prgm = MSCCLPPProgram("allreduce", topology, collective, 1)
    # The dependency graph for rank 1 is as follows:
    # put(0i to 1s) => reduce(1s to 1i) => put(2i to 1s) => reduce(1s to 1i)
    #                       |  => put(1i to 0s)                      ^
    #                       |  =>  put(1i to 2s)------------------- -|
    # put(2i to 1s) => reduce(1s to 1i) for read after write
    # put(1i to 2s) => reduce(1s to 1i) for write after read
    # when we try to merge reduce(1s to 1i) => put(2i to 1s) => reduce(1s to 1i),
    # circular dependency is introduced
    with prgm:
        c0 = chunk(0, Buffer.input, 0)
        c0.put_packet(1, "scratch", 0, sendtb=0)
        c1s = chunk(1, "scratch", 0)
        c1 = chunk(1, Buffer.input, 0)
        c1 = c1.reduce_packet(c1s, recvtb=0)
        c1.put_packet(0, "scratch", 0, sendtb=0)
        c1.put_packet(2, "scratch", 0, sendtb=0)
        c2 = chunk(2, Buffer.input, 0)
        c2.put_packet(1, "scratch", 0, sendtb=0)
        c1.reduce_packet(c1s, recvtb=0)
    lowered_prgm = prgm.lower()
    lowered_prgm.gpus[1].threadblocks = [tb for tb in lowered_prgm.gpus[1].threadblocks if tb.id != -1]
    assert lowered_prgm.gpus[1].threadblocks[0].ops[0].inst == MscclppInstruction.reduce_send_packet
    assert lowered_prgm.gpus[1].threadblocks[0].ops[1].inst == MscclppInstruction.reduce_packet


def test_replication():
    topology = fully_connected(2)
    collective = AllToAll(2, 1, False)
    prgm = MSCCLProgram("alltoall", topology, collective, 1)
    with prgm:
        chunk(0, Buffer.input, 0).copy(0, Buffer.output, 0)
        chunk(0, Buffer.input, 1).copy(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).copy(0, Buffer.output, 1)
        chunk(1, Buffer.input, 1).copy(1, Buffer.output, 1)

    instances = 2
    replicated_prgm = MSCCLProgram("alltoall", topology, collective, instances)
    with replicated_prgm:
        chunk(0, Buffer.input, 0).copy(0, Buffer.output, 0)
        chunk(0, Buffer.input, 1).copy(1, Buffer.output, 0)
        chunk(1, Buffer.input, 0).copy(0, Buffer.output, 1)
        chunk(1, Buffer.input, 1).copy(1, Buffer.output, 1)

    lowered_prgm = prgm.lower()
    lowered_replicated_prgm = replicated_prgm.lower()

    for gpu1, gpu2 in zip(lowered_prgm.gpus, lowered_replicated_prgm.gpus):
        assert len(gpu1.threadblocks) * instances == len(gpu2.threadblocks)


def test_illegal_tb_assignment():
    num_gpus = 3
    topology = fully_connected(num_gpus)
    collective = AllToAll(num_gpus, 1, False)
    prgm = MSCCLProgram("alltoall", topology, collective, 1, threadblock_policy=ThreadblockPolicy.manual)
    with prgm:
        with pytest.raises(Exception):
            # Cannot send to two different gpus on the same threadblock
            chunk(0, Buffer.input, 0).copy(1, Buffer.output, 0, sendtb=0, recvtb=1)
            chunk(0, Buffer.input, 1).copy(2, Buffer.output, 0, sendtb=0, recvtb=1)
            XML()


def test_group_api():
    num_gpus = 4
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)
    prgm = MSCCLPPProgram("allreduce", topology, collective, 1)
    with prgm:
        for rank in range(num_gpus):
            index = rank
            reduce_chunks = []
            c = chunk(rank, Buffer.input, index)
            # make sure the data is ready
            for nghr in range(num_gpus):
                if rank != nghr:
                    c_peer = chunk(nghr, Buffer.input, index)
                    reduce_chunks.append(c_peer)
            c = c.group_load_reduce(reduce_chunks, recvtb=0)
            ngbrs = [nghr for nghr in range(num_gpus) if nghr != rank]
            c.group_store(ngbrs, sendtb=0)
        assert Check()
    lowered_prgm = prgm.lower()
    for gpu in lowered_prgm.gpus:
        for tb in gpu.threadblocks:
            assert len(tb.ops) == 1
            assert tb.ops[0].inst == MscclppInstruction.group_load_reduce_store


def test_routines_allgather_ring_inplace():
    size = 4
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram("allgather_ring", topology, collective, 1):
        allgather_ring_inplace(size)
        assert Check()


def test_routines_allgather_ring_nodes():
    size = 8
    topology = fully_connected(size)
    collective = AllGather(size, 1, True)
    with MSCCLProgram("allgather_multi", topology, collective, 1):
        # Two parallel rings [0-4] and [4-8]
        allgather_ring_inplace(4, 0, 0)
        allgather_ring_inplace(4, 4, 4)
        # Exchange between peers (0,4) (1,5) etc.
        for r in range(0, 8):
            peer = (r + 4) % size
            exchange_index = 0 if r < 4 else 4
            chunk(r, Buffer.output, exchange_index, 4).copy(peer, Buffer.output, exchange_index)
        assert Check()


def test_routines_allreduce_ring_inplace():
    size = 4
    topology = fully_connected(size)
    collective = AllReduce(size, size, True)
    with MSCCLProgram("allreduce_ring", topology, collective, 1):
        allreduce_ring_inplace(size)
        assert Check()


def test_routines_allreduce_nodes():
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size, True)
    with MSCCLProgram("allreduce_multi", topology, collective, 1):
        # Two parallel rings [0-4] and [4-8]
        allreduce_ring_inplace(4, 0, 0)
        allreduce_ring_inplace(4, 0, 4, ch=1)

        allreduce_ring_inplace(4, 4, 4)
        allreduce_ring_inplace(4, 4, 0, ch=1)
        # Reduction between peers (0,4) (1,5) etc.
        for r in range(0, 8):
            peer = (r + 4) % size
            exchange_index = 0 if r < 4 else 4
            c = chunk(peer, Buffer.output, exchange_index, 4)
            c.reduce(chunk(r, Buffer.output, exchange_index, 4))
            c = c.copy(r, Buffer.output, exchange_index)
        XML()
        assert Check()


def test_routines_allreduce_packet_inplace_mscclpp():
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * size, True)
    with MSCCLPPProgram("allreduce_packet", topology, collective, 2, protocol="LL"):
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for tb in range(size):
                if tb == r1:
                    continue
                remote_rank = tb
                index = remote_rank * size
                c = chunk(r1, Buffer.input, index, size)
                c.put_packet(remote_rank, "scratch", index=r1 * size, sendtb=tb)
        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(size):
                c = chunk(r, Buffer.input, r * size + index)
                for peer in range(size):
                    if peer != r:
                        c.reduce_packet(chunk(r, "scratch", peer * size + index), recvtb=index)
                for peer in range(size):
                    if peer != r:
                        c.put_packet(peer, "scratch", (size * size) + r * size + index, sendtb=index)
        # Each rank get final result from scratch space
        for r in range(size):
            for peer in range(size):
                if peer != r:
                    c = chunk(r, "scratch", size * size + peer * size, size)
                    c.copy_packet(r, Buffer.input, peer * size, sendtb=peer)
        Json()
        assert Check()

def test_routines_allreduce_inplace_mscclpp():
    size = 8
    topology = fully_connected(size)
    collective = AllReduce(size, size * size, True)
    with MSCCLPPProgram("allreduce_pairs", topology, collective, 2, protocol="Simple"):
        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            for tb in range(size):
                index = rank * size
                c = chunk(rank, Buffer.input, index + tb)
                # make sure the data is ready
                for nghr in range(size):
                    peer_index = nghr * size
                    if rank != nghr:
                        c_peer = chunk(rank, Buffer.input, peer_index + tb)
                        c_peer.signal(nghr, Buffer.input, peer_index + tb, sendtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.wait(nghr, Buffer.input, index + tb, recvtb=tb)
                # reduce the chunks
                for i in range(size):
                    nghr = (rank + i) % size
                    if rank != nghr:
                        c.reduce(chunk(nghr, Buffer.input, index + tb), recvtb=tb)
                for nghr in range(size):
                    if rank != nghr:
                        c.signal(nghr, Buffer.input, index + tb, sendtb=tb)

        # wait for all the chunks is ready, then get the chunks
        for rank in range(size):
            for tb in range(size):
                for nghr in range(size):
                    if rank != nghr:
                        index = nghr * size
                        c = chunk(rank, Buffer.input, index + tb)
                        c.wait(nghr, Buffer.input, index + tb, recvtb=tb)
                for i in range(size):
                    nghr = (rank + i) % size
                    index = nghr * size
                    if rank != nghr:
                        c = chunk(rank, Buffer.input, index + tb)
                        c.get(nghr, Buffer.input, index + tb, recvtb=tb)

        Json()
        assert Check()
