# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language.buffer import *
from msccl.language.instruction_dag import *
from msccl.language.passes import *
from msccl.language.tb_assignment import *
from msccl.language.types import ThreadblockPolicy

_current_program = None


def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program


class MSCCLProgram:
    def __init__(
        self,
        name,
        topo,
        collective,
        instances,
        protocol="Simple",
        threadblock_policy=ThreadblockPolicy.auto,
        interleaved_replication=True,
        instr_fusion=True,
        check_xml=True,
        dependence_nop=False,
    ):
        self.name = name
        self.topo = topo
        self.collective = collective
        self.num_ranks = topo.num_nodes()
        self.instances = instances
        self.protocol = protocol
        self.threadblock_policy = threadblock_policy
        self.interleaved_replication = interleaved_replication
        self.instr_fusion = instr_fusion
        self.check_xml = check_xml
        self.dependence_nop = dependence_nop
        assert (
            protocol == "Simple" or protocol == "LL" or protocol == "LL128"
        ), f"Given protocol: {protocol}. Must be either Simple, LL, LL128"
        self.run_opt = True  # Runs optimization passes
        # Initialize the input buffers
        # self.chunk_dag = ChunkDAG()
        self.buffers = collective.init_buffers()
        self.instr_dag = MscclInstructionDAG(self.num_ranks, self.buffers)
        for r in range(self.num_ranks):
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                ref = self.get_ref(r, buffer, index, 1)
                # self.chunk_dag.init_chunk(chunk, ref)
                self.instr_dag.add_start(r, buffer, index, ref)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCL Program in context")
        _current_program = self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

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

    # Lower program to XML
    def lower(self):
        # self.chunk_dag._complete_metadata()
        # self.chunk_dag.channel_assignment()
        # self.chunk_dag.lower_instr_dag(self.instr_dag)
        self.instr_dag.convert_set_list()  # Pre-emptively convert sets to lists
        if self.instr_fusion:
            self.instr_dag.optimize()
        self.instr_dag._complete_metadata()
        if self.threadblock_policy == ThreadblockPolicy.manual:
            manual_assign_tbs(self.instr_dag)
        else:
            auto_assign_tbs(self.instr_dag)
        self.instr_dag.lower_pt1(self.instances)
        gpu_prgms = self.instr_dag.lower_pt2(self.instances, self.interleaved_replication)
        if self.check_xml:
            # Check generated MSCCL-IR for correctness - no circular dependencies, sends and receives are ordered
            # For very large programs, turn off check_xml when shipping
            check_dependency_cycles(self.instr_dag.tbs)
            check_threadblock_ordering(self.instr_dag)
        return Program(self.name, self.collective.name, self.collective.inplace, self.protocol, gpu_prgms)

    def generate_xml(self):
        return ir_to_xml(self.lower(), dependence_nop=self.dependence_nop)

    def print_chunk_dag(self):
        visualize_chunk_dag(self.chunk_dag.chunk_paths)

    def print_instr_dags(self, rank):
        if rank == 0:
            for r in range(len(self.ranks)):
                visualize_instr_dag(self.instr_dags[r].operations)
        else:
            visualize_instr_dag(self.instr_dags[rank].operations)


def XML():
    print(_curr().generate_xml())


@dataclass
class Ref(ChunkRef):
    prog: MSCCLProgram

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

    # Copies the chunk(s) referenced by this chunkref onto Rank dst at location (buffer, index)
    def copy(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=-1):
        self.prog.check_buffer_exists(dst, buffer)

        # If index is not specified assume it is going to the same place in the next gpu
        if index == -1 and buffer == None:
            index = self.index
            buffer = self.buffer
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            index = self.prog.buffers[dst][buffer].instance_size()

        # Some inplace collectives have custom logic for buffers and index (ReduceScatter, AllGather)
        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)

        # Direct send
        assert self.prog.topo.link(self.rank, dst) or dst == self.rank, f"No link from {self.rank} to {dst}"
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        # Check if we are copying the chunk to the same index (easy mistake when we are using inplace)
        if dst_chunkref == self:
            return

        # chunks = self.prog.get_chunks(self.rank, self.buffer, self.index, self.size)
        # overwritten_chunks = self.prog.get_chunks(dst, buffer, index, self.size)

        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)

        # self.prog.chunk_dag.add_send(chunks, overwritten_chunks, self, dst_chunkref, sendtb, recvtb, ch)
        sender = self.rank
        receiver = dst
        if sender != receiver:
            sop = self.prog.instr_dag.add_send(sender, self, dst_chunkref, sendtb, ch)
            rop = self.prog.instr_dag.add_recv(receiver, self, dst_chunkref, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_copy(sender, self, dst_chunkref, sendtb, ch)

        return dst_chunkref

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce(self, other_chunkref, sendtb=-1, recvtb=-1, ch=-1):
        # Receive reduce copy
        dst = self.rank
        src = other_chunkref.rank
        assert self.prog.topo.link(src, dst) or src == dst, f"No link from {src} to {dst}"
        # dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        # chunks1 = self.prog.get_chunks(self.rank, self.buffer, self.index, self.size)
        # chunks2 = self.prog.get_chunks(other_chunkref.rank, other_chunkref.buffer, other_chunkref.index self.size)

        self.prog.apply_reduce(
            src, other_chunkref.buffer, other_chunkref.index, dst, self.buffer, self.index, self.size
        )

        # reduce_chunks = self.prog.get_chunks(dst, buffer, index, self.size)
        # self.prog.chunk_dag.add_reduce(chunks1, chunks2, reduce_chunks, self, dst_chunkref, sendtb, recvtb, ch)
        if src != dst:
            sop = self.prog.instr_dag.add_send(src, other_chunkref, self, sendtb, ch)
            rop = self.prog.instr_dag.add_recv_reduce_copy(dst, other_chunkref, self, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_reduce(src, other_chunkref, self, sendtb, ch)

        return self

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
