# Copyright (c) 2024 Advanced Micro Devices.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

# Implementation of ring-based AllGather where data is being pushed using put()
def allgather_ring_push(size, in_place):
    topology = fully_connected(size)
    collective = AllGather(size, 1, in_place)
    with MSCCLPPProgram(f"allgather_ring_push", topology, collective, 1):
        # If not in-place copy local data chunk to output buffer
        if not in_place:
            for rank in range(0, size):
                c = chunk(rank, Buffer.input, 0)
                c.copy(rank, Buffer.output, rank)
        # Iterate over steps
        for step in range(0, size - 1):
            for rank in range(0, size):
                # Put & Signal
                index = (rank - step) % size
                c = chunk(rank, Buffer.output, index)
                next_rank = (rank + 1) % size
                c.put(next_rank, Buffer.output, index, sendtb=0) # TODO how does this guarantee that the buffer is ready?
                c.signal(next_rank, Buffer.output, index, 0)
                # Wait
                prev_rank = (rank - 1) % size
                recv_index = (rank - step - 1) % size
                c = chunk(rank, Buffer.output, recv_index)
                c.wait(prev_rank, Buffer.output, recv_index, 0)
        Json()
        Check()

# Implementation of ring-based AllGather where data is being pulled using get()
def allgather_ring_pull(size, in_place):
    topology = fully_connected(size)
    collective = AllGather(size, 1, in_place)
    with MSCCLPPProgram(f"allgather_ring_pull", topology, collective, 1):
        # If not in-place copy local data chunk to output buffer
        if not in_place:
            for rank in range(0, size):
                c = chunk(rank, Buffer.input, 0)
                c.copy(rank, Buffer.output, rank)
        # Iterate over steps
        for step in range(0, size - 1): # size - 1):
            for rank in range(0, size):
                # Signal
                index = (rank - step) % size
                c = chunk(rank, Buffer.output, index)
                next_rank = (rank + 1) % size
                c.signal(next_rank, Buffer.output, index, 0)
                # Wait & Get
                prev_rank = (rank - 1) % size
                recv_index = (rank - step - 1) % size
                c = chunk(rank, Buffer.output, recv_index)
                c.wait(prev_rank, Buffer.output, recv_index, 0)
                c.get(prev_rank, Buffer.output, recv_index, recvtb=0)
        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('--in-place', type=bool, default=True, help='Do collective in-place?')

args = parser.parse_args()

# allgather_ring_push(args.num_gpus, args.in_place)
allgather_ring_pull(args.num_gpus, args.in_place)