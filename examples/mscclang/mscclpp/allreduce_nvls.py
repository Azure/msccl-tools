# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


def allreduce_allpairs(gpus, instances):
    size = gpus
    chunksperloop = gpus
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLPPProgram(
        "allreduce_nvls",
        topology,
        collective,
        instances,
    ):
        # Each rank sends the nth chunk to the nth rank into scratch space
        for rank in range(size):
            index = rank
            c = chunk(rank, Buffer.input, index)
            reduce_chunks = []
            # make sure the data is ready
            for nghr in range(size):
                if rank != nghr:
                    c_peer = chunk(nghr, Buffer.input, index)
                    reduce_chunks.append(c_peer)
                    c.signal(nghr, Buffer.input, index, sendtb=0)
            for nghr in range(size):
                if rank != nghr:
                    c.wait(nghr, Buffer.input, index, recvtb=0)
            c.group_load_reduce(reduce_chunks, recvtb=0)
            ngbrs = [nghr for nghr in range(size) if nghr != rank]
            c.group_store(ngbrs, sendtb=0)

        Json()
        # Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances)
