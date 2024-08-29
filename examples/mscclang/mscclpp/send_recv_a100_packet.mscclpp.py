# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import SendRecv


def send_recv(instances):
    size = 2
    chunksperloop = 1
    topology = fully_connected(size)
    collective = SendRecv(size, chunksperloop, False)
    with MSCCLPPProgram(
        "send_recv",
        topology,
        collective,
        instances,
        protocol="LL",
    ):
        for r in range(size):
            for nghr in range(size):
                if nghr == r:
                    continue
                c = chunk(r, Buffer.input, 0)
                c.put_packet(nghr, "scratch", 0, sendtb=0, channel_type=ChannelType.proxy)

        for r in range(size):
            for nghr in range(size):
                if nghr == r:
                    continue
                c = chunk(r, "scratch", 0)
                c.get_packet(nghr, Buffer.output, 0, recvtb=0, channel_type=ChannelType.proxy)

        Json()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

send_recv(args.instances)
