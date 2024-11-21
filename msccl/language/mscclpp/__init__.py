# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.collectives import Collective
from msccl.language.buffer import *
from msccl.language.types import ChannelType
from msccl.language.mscclpp.ir import *
from msccl.language.mscclpp.instruction_dag import MscclppInstructionDAG
from msccl.language.tb_assignment import *
from msccl.topologies.topology import Topology

_current_program = None


def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program


# For msccl++ program, we have one assumption that for channel can be identified by (send_buffer, recv_buffer, type, send_tb/recv_tb)
# which means the send_tb and recv_tb should be the same for a pair of signal and wait, also same for put/get operation.
# If one sender what to send data to peer want to use different tb in receiver side. We need to send to same tb in receiver side first,
# then performance a across tb sync. This is a limitation of current implementation.
class MSCCLPPProgram:
    def __init__(
        self,
        name: str,
        topo: Topology,
        collective: Collective,
        instances: int,
        protocol: str = "Simple",
        instr_fusion: bool = True,
        replication_policy: ReplicationPolicy = ReplicationPolicy.duplicated,
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
    ):
        self.name = name
        self.topo = topo
        self.collective = collective
        self.num_ranks = topo.num_nodes()
        self.instances = instances
        self.protocol = protocol
        self.instr_fusion = instr_fusion
        self.replication_policy = replication_policy
        self.num_threads_per_block = num_threads_per_block
        self.use_double_scratch_buffer = use_double_scratch_buffer
        assert protocol == "Simple" or protocol == "LL", f"Given protocol: {protocol}. Must be either Simple, LL"
        self.run_opt = True  # Runs optimization passes
        # Initialize the input buffers
        self.buffers = collective.init_buffers()
        self.instr_dag = MscclppInstructionDAG(self.num_ranks, self.buffers)
        for r in range(self.num_ranks):
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                ref = self.get_ref(r, buffer, index, 1)
                # self.chunk_dag.init_chunk(chunk, ref)
                self.instr_dag.add_start(r, buffer, index, ref)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCLPP Program in context")
        _current_program = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

    def _convert_to_exectuion_plan(self):
        ops = self.instr_dag.convert_set_list()
        ops = sorted(ops, key=lambda x: x.step)
        for op in ops:
            rank = op.rank
            tbid = op.tb
            if tbid not in self.instr_dag.tbs[rank]:
                self.instr_dag.tbs[rank][tbid] = Threadblock(id=tbid)
            tb = self.instr_dag.tbs[rank][tbid]
            tb.ops.append(op)

    def get_rank_ref(self, rank):
        return RankRef(rank, self)

    # Tracks a send operation on the buffers
    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

    # Tracks a reduce operation on the buffers
    def apply_reduce(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            reduce_chunk = db[dst_index + i]
            sent_chunk = sb[src_index + i]
            db[dst_index + i] = reduce_chunk.reduce(dst, sent_chunk)

    def get_ref(self, rank, buffer, index, size):
        buffer, index = self.collective.get_buffer_index(rank, buffer, index)
        return Ref(rank, buffer, index, size, self)

    def get_chunks(self, rank, buffer, index, size=1):
        chunks = [None] * size
        for i in range(0, size):
            if self.buffers[rank][buffer] and index + i < len(self.buffers[rank][buffer]):
                chunks[i] = self.buffers[rank][buffer][index + i]
            else:
                chunks[i] = None
        return chunks

    def check_buffer_exists(self, rank, name):
        if name not in self.buffers[rank]:
            self.buffers[rank][name] = BufferSlice(Buffer.scratch, name)

    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def check(self):
        return self.collective.check(self)

    # Lower program to MSCCLPP
    def lower(self):
        self._convert_to_exectuion_plan()
        self.instr_dag.complete_channels()
        self.instr_dag.remove_redundant_signal_wait()
        if self.instr_fusion:
            self.instr_dag.optimize()
        self.instr_dag.lower_pt1(self.instances)
        gpu_prgms = self.instr_dag.lower_pt2(self.instances, self.replication_policy)
        return Program(
            self.name,
            self.collective.name,
            self.collective.inplace,
            self.protocol,
            gpu_prgms,
            self.collective.num_chunk_groups * self.instances,
            self.num_threads_per_block,
            self.use_double_scratch_buffer,
        )

    def generate_json(self):
        return ir_to_json(self.lower())


def Json():
    print(_curr().generate_json())


@dataclass
class RankRef:
    rank: int
    prog: MSCCLPPProgram

    def barrier(self, tb_list):
        return self.prog.instr_dag.add_barrier(self.rank, tb_list)


@dataclass
class Ref(ChunkRef):
    prog: MSCCLPPProgram

    def __repr__(self):
        return f"Ref(Buffer:{self.buffer}, Index:{self.index}, Size:{self.size}, Rank:{self.rank})"

    def _end(self):
        return self.index + self.size

    def _get_chunk(self, index):
        return self.prog.buffers[self.rank][self.buffer][index]

    def split(self, num):
        assert self.size % num == 0, f"Trying to split a chunk of {self.size} elements into {num} parts"
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = self.prog.get_ref(self.rank, self.buffer, index, size)
        return chunks

    def group(self, other):
        assert self.rank == other.rank, f"Trying to concatenate chunks on ranks {self.rank} and {other.rank}"
        assert self.buffer == other.buffer, f"Trying to concatenate chunks in {self.buffer} and {other.buffer}"
        if self.index < other.index:
            first = self
            second = other
        else:
            first = other
            second = self

        end = max(first._end(), second._end())
        return Ref(self.rank, self.buffer, first.index, end - first.index, self.prog)

    def _get_buffer_index(self, remote_rank, buffer, index):
        if index == -1 and buffer == None:
            return self.buffer, self.index
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            return buffer, self.prog.buffers[remote_rank][buffer].instance_size()
        return buffer, index

    def _put(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.sm, use_packet=False):
        self.prog.check_buffer_exists(dst, buffer)
        assert self.rank != dst, "Cannot put to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        # Direct put
        assert self.prog.topo.link(self.rank, dst) or dst == self.rank, f"No link from {self.rank} to {dst}"
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)
        if use_packet:
            self.prog.instr_dag.add_put(self.rank, self, dst_chunkref, sendtb, chan_type, True)
            self.prog.instr_dag.add_signal(self.rank, self, dst_chunkref, -1, ChannelType.none)
            self.prog.instr_dag.add_wait(dst, dst_chunkref, self, -1, ChannelType.none)
        else:
            self.prog.instr_dag.add_put(self.rank, self, dst_chunkref, sendtb, chan_type)
        return dst_chunkref

    def put(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.sm):
        return self._put(dst, buffer, index, sendtb, chan_type)

    def put_packet(
        self,
        dst,
        buffer=None,
        index=-1,
        sendtb=-1,
        chan_type=ChannelType.sm,
        temp_buffer=None,
        temp_buffer_index=-1,
    ):
        chunk_ref = self
        if chan_type == ChannelType.proxy:
            assert temp_buffer is not None, "Need to specify a temporary buffer for proxy channels"
            chunk_ref = self._copy(
                self.rank, temp_buffer, temp_buffer_index, sendtb, trans_from_packet=False, trans_to_packet=True
            )
        return chunk_ref._put(dst, buffer, index, sendtb, chan_type, True)

    def get(self, src, buffer=None, index=-1, recvtb=-1, chan_type=ChannelType.sm):
        self.prog.check_buffer_exists(src, buffer)
        sender = src
        receiver = self.rank
        assert sender != receiver, "Cannot get from the same rank"
        buffer, index = self._get_buffer_index(src, buffer, index)

        # Direct get
        assert self.prog.topo.link(self.rank, src) or src == self.rank, f"No link from {self.rank} to {src}"
        src_chunkref = self.prog.get_ref(src, buffer, index, self.size)

        self.prog.apply_send(src, buffer, index, self.rank, self.buffer, self.index, self.size)
        self.prog.instr_dag.add_get(receiver, src_chunkref, self, recvtb, chan_type)

    # for signal and wait, currently we assuem the pair will use the same tb index. In future we need
    # to infer the tb index from the instruction DAG Add a channel is define as (send_tb, src_buffer, recv_tb, dst_buffer, type).
    # Then we can use DAG info to reduce the number of channels.
    def signal(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.sm):
        sender = self.rank
        receiver = dst
        assert sender != receiver, "Cannot signal to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        # Direct signal
        assert self.prog.topo.link(self.rank, dst) or dst == self.rank, f"No link from {self.rank} to {dst}"
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        self.prog.instr_dag.add_signal(sender, self, dst_chunkref, sendtb, chan_type)

    # only proxy channel need to use this function
    def flush(self, dst, buffer=None, index=-1, sendtb=-1, chan_type=ChannelType.proxy):
        assert chan_type == ChannelType.proxy, "Only proxy channel can use flush"
        sender = self.rank
        receiver = dst
        assert sender != receiver, "Cannot flush to the same rank"
        buffer, index = self._get_buffer_index(dst, buffer, index)

        assert self.prog.topo.link(self.rank, dst) or dst == self.rank, f"No link from {self.rank} to {dst}"
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        self.prog.instr_dag.add_flush(sender, self, dst_chunkref, sendtb)

    def wait(self, src, buffer=None, index=-1, recvtb=-1, chan_type=ChannelType.sm):
        sender = src
        receiver = self.rank
        assert sender != receiver, "Cannot wait on the same rank"
        buffer, index = self._get_buffer_index(src, buffer, index)

        # Direct wait
        assert self.prog.topo.link(self.rank, src) or src == self.rank, f"No link from {self.rank} to {src}"
        src_chunkref = self.prog.get_ref(src, buffer, index, self.size)
        self.prog.instr_dag.add_wait(receiver, self, src_chunkref, recvtb, chan_type)

    def _copy(self, dst, buffer=None, index=-1, sendtb=-1, trans_from_packet=False, trans_to_packet=False):
        self.prog.check_buffer_exists(dst, buffer)
        buffer, index = self._get_buffer_index(dst, buffer, index)

        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)
        # Check if we are copying the chunk to the same index (easy mistake when we are using inplace)
        if dst_chunkref == self:
            return
        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)

        assert self.rank == dst, "Chunk copy only supports intra-rank communication"
        self.prog.instr_dag.add_copy(self.rank, self, dst_chunkref, sendtb, trans_from_packet, trans_to_packet)

        return dst_chunkref

    # Copies the chunk(s) referenced by this chunkref onto Rank dst at location (buffer, index)
    def copy(self, dst, buffer=None, index=-1, sendtb=-1):
        return self._copy(dst, buffer, index, sendtb)

    def copy_packet(self, dst, buffer=None, index=-1, sendtb=-1):
        return self._copy(dst, buffer, index, sendtb, trans_from_packet=True, trans_to_packet=False)

    def _reduce(self, other_chunkref, recvtb=-1, channel_type=ChannelType.sm, use_packet=False):
        dst = self.rank
        src = other_chunkref.rank
        assert self.prog.topo.link(src, dst) or src == dst, f"No link from {src} to {dst}"
        self.prog.apply_reduce(src, other_chunkref.buffer, other_chunkref.index, dst, self.buffer, self.index, self.size)
        if use_packet:
            assert src == dst, "Packet reduce only supports intra-rank communication"

        if src != dst:
            self.prog.instr_dag.add_read_reduce(dst, other_chunkref, self, recvtb, channel_type)
        else:
            self.prog.instr_dag.add_reduce(src, other_chunkref, self, recvtb, use_packet)

        return self

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce(self, other_chunkref, recvtb=-1, channel_type=ChannelType.sm):
        return self._reduce(other_chunkref, recvtb, channel_type)

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce_packet(self, other_chunkref, recvtb=-1):
        return self._reduce(other_chunkref, recvtb, use_packet=True)

    def get_origin_index(self, index=0):
        return self._get_chunk(index + self.index).origin_index

    def get_origin_rank(self, index=0):
        return self._get_chunk(index + self.index).origin_rank

    def get_dst_index(self, index=0):
        return self._get_chunk(index + self.index).dst_index

    def get_dst_rank(self, index=0):
        return self._get_chunk(index + self.index).dst_rank

    def print_chunk_info(self, index=0):
        print(self._get_chunk(index + self.index))
