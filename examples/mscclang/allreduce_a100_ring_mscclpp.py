# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


# Ring all reduce for A100s
def allreduce_ring(size, instances):
    topology = fully_connected(size)
    collective = AllReduce(size, size, True)
    with MSCCLPPProgram(
        f"allreduce_ring",
        topology,
        collective,
        instances,
        protocol="Simple",
    ):
        # Reduce ring
        for step in range(0, size - 1):
            for index in range(0, size):
                rank = (index + step) % size
                next_rank = (index + step + 1) % size
                c = chunk(rank, Buffer.input, index)
                c.signal(next_rank, Buffer.input, index, 0)
                prev_rank = (index + step - 1) % size
                c = chunk(rank, Buffer.input, (index + size - 1) % size)
                c.wait(prev_rank, Buffer.input, (index + size - 1) % size, 0)
                c.reduce(chunk(prev_rank, Buffer.input, (index + size - 1) % size), recvtb=0)

        # Propagate ring
        for step in range(-1, size - 2):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                next_rank = (index + step + 1) % size
                c.put(next_rank, Buffer.input, index, sendtb=0)
                c.signal(next_rank, Buffer.input, index, 0)
                prev_rank = (index + step - 1) % size
                c = chunk(rank, Buffer.input, (index + size - 1) % size)
                c.wait(prev_rank, Buffer.input, (index + size - 1) % size, 0)

        Json()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")
args = parser.parse_args()

allreduce_ring(args.num_gpus, args.instances)
