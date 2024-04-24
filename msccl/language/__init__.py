# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language.ir import *
from msccl.language.passes import *
from msccl.language.tb_assignment import *
from msccl.language.chunk import *
from msccl.language.buffer import *
from msccl.language.instruction_dag import *
import msccl.language.msccl as msccl_lang
import msccl.language.mscclpp as mscclpp
from msccl.language.mscclpp import *
from msccl.language.msccl import *
from typing import Union

# from msccl.language.visualize import *


def _curr():
    if msccl_lang._current_program == None and mscclpp._current_program == None:
        raise RuntimeError("No Program in context")
    if msccl_lang._current_program == None:
        return mscclpp._current_program
    return msccl_lang._current_program


def Print():
    _curr().print_chunk_dag()


def chunk(rank, buffer, index, size=1) -> Union[mscclpp.Ref, msccl_lang.Ref]:
    if _curr().buffers[rank][buffer][index] is None:
        return None
    return _curr().get_ref(rank, buffer, index, size)


def create_scratch(rank, name):
    return _curr().create_scratch(rank, name)


def Check():
    return _curr().check()


# @dataclass
# class ChunkOp():
#     inst: ChunkInstruction
#     src: Ref # Ref Chunk acted on
#     dst: Ref # Ref Chunk created
#     sendtb: int = -1# For lowering to RankInstructions
#     recvtb: int = -1#  For lowering to RankInstructions
#     ch: int = -1 # For lowering to RankInstructions
#     steps_from_start:int  = -1
#     steps_to_end: int = -1
#     prev: list = field(default_factory=list) # Previous ChunkOps
#     next: list = field(default_factory=list) # Next ChunkOps
#     visited = False
#     num = -1

#     def __repr__(self):
#         return f'ChunkOp({self.inst} {self.dst.rank} {self.dst.buffer} {self.dst.index})'

#     def __lt__(self, other):
#         return self.steps_from_start < other.steps_from_start

#     def __hash__(self):
#         return hash((self.inst, self.dst.rank, self.dst.index, self.dst.buffer)) # TODO

# def same_slot(ref1, ref2):
#     return ref1.rank == ref2.rank and ref1.buffer == ref2.buffer and ref1.index == ref2.index

# # Returns if there is overlap between the refs
# def overlap_refs(ref1, ref2):
#     same_location = ref1.rank == ref2.rank and ref1.buffer == ref2.buffer
#     if same_location:
#         ref1_range = (ref1.index, ref1.index + ref1.size)
#         ref2_range = (ref2.index, ref2.index + ref2.size)
#         if ref1_range < ref2_range:
#             return ref1_range[0] < ref2_range[1]
#         else:
#             return ref2_range[0] < ref1_range[1]
#     return False

# class ChunkDAG:

#     def __init__(self):
#         self.chunks = []
#         self.chunk_paths = {} # chunk -> ChunkOp. Stores the entry point to where every chunk is created
#         self.max_hops = -1

#     # Initialize the ChunkDAG with starting chunks
#     def init_chunk(self, chunk, ref):
#         op = ChunkOp(ChunkInstruction.start, None, ref, steps_from_start=-1)
#         self.chunks.append(chunk)
#         self.chunk_paths[chunk] = op

#     def _find_prev_op_for_chunk(self, chunk, ref):
#         prev_op = None
#         frontier = [self.chunk_paths[chunk]]
#         while len(frontier) > 0:
#             current_op = frontier[0]
#             if overlap_refs(ref, current_op.dst):
#                 prev_op = current_op
#             frontier = frontier[1:] + current_op.next
#         return prev_op

#     def add_send(self, chunks, overwritten_chunks, src, dst, sendtb, recvtb, ch):
#         # Find the previous operation for these chunks
#         prev_ops = []
#         steps_from_start = 0
#         for chunk1, chunk2 in zip(chunks, overwritten_chunks):
#             prev_op_src = self._find_prev_op_for_chunk(chunk1, src)
#             if chunk2 is None:
#                 steps_from_start = max(steps_from_start, prev_op_src.steps_from_start)
#             else:
#                 prev_op_dst = self._find_prev_op_for_chunk(chunk2, dst) # In case we overwrite
#                 steps_from_start = max(prev_op_src.steps_from_start, prev_op_dst.steps_from_start, steps_from_start)
#                 prev_ops.append(prev_op_dst)
#             prev_ops.append(prev_op_src)
#             # prev_op = self._find_prev_op_for_chunk(chunk, src)
#             # steps_from_start = max(steps_from_start, prev_op.steps_from_start)
#             # prev_ops.append(prev_op)
#         op = ChunkOp(ChunkInstruction.send, src, dst, sendtb, recvtb, ch, steps_from_start+1)

#         for prev_op in prev_ops:
#             prev_op.next.append(op)
#         op.prev = prev_ops

#     def add_reduce(self, chunks1, chunks2, reduce_chunks, src, dst, sendtb, recvtb, ch):
#         # self.chunks.append(reduce_chunks)
#         prev_ops = []
#         steps_from_start = 0
#         # Find the previous operations that reduce builds off
#         for chunk1, chunk2 in zip(chunks1, chunks2):
#             prev_op_src = self._find_prev_op_for_chunk(chunk1, src)
#             prev_op_dst = self._find_prev_op_for_chunk(chunk2, dst)
#             steps_from_start = max(prev_op_src.steps_from_start, prev_op_dst.steps_from_start, steps_from_start)
#             prev_ops.append(prev_op_src)
#             prev_ops.append(prev_op_dst)

#         op = ChunkOp(ChunkInstruction.reduce, src, dst, sendtb, recvtb, ch, steps_from_start+1)

#         for prev_op in prev_ops:
#             prev_op.next.append(op)
#             op.prev.append(prev_op)

#         # Reduce operations create new chunks, so keep a pointer to a new chunk
#         for rc in reduce_chunks:
#             self.chunk_paths[rc] = op

#     def _complete_metadata(self):
#         def dfs(op):
#             if len(op.next) == 0:
#                 op.steps_to_end = 0
#             else:
#                 for o in op.next:
#                     dfs(o)
#                 op.steps_to_end = functools.reduce(lambda cur, x: max(cur, x.steps_to_end+1), op.next, 0)

#         for chunk, op in self.chunk_paths.items():
#             if op.inst == ChunkInstruction.start:
#                 dfs(op)


#     # Assigns each send and a reduce a channel for communication based of policies
#     def channel_assignment(self, channel_policy='zero'):
#         frontier = []
#         visited = set()
#         for chunk, op in self.chunk_paths.items():
#             if len(op.prev) == 0:
#                 heapq.heappush(frontier, op)

#         # If an op isn't annotated with a channel set it to 0
#         if channel_policy == 'zero':
#             while len(frontier) > 0:
#                 op = heapq.heappop(frontier)
#                 if op not in visited:
#                     op.ch = 0 if op.ch == -1 else op.ch
#                     for o in op.next:
#                         heapq.heappush(frontier, o)
#                     visited.add(op)

#     def lower_instr_dag(self, instr_dag):
#         frontier = []
#         visited = set()

#         for chunk, op in self.chunk_paths.items():
#             if len(op.prev) == 0:
#                 heapq.heappush(frontier, ((op.steps_from_start, op.steps_to_end), op))

#         while len(frontier) > 0:
#             _, op = heapq.heappop(frontier)
#             if op not in visited:
#                 sendtb = op.sendtb
#                 recvtb = op.recvtb
#                 ch =  op.ch
#                 if op.inst == ChunkInstruction.start:
#                     rank = op.dst.rank
#                     instr_dag.add_start(rank, op.dst.buffer, op.dst.index, op.dst)
#                 elif op.inst == ChunkInstruction.send:
#                     sender = op.src.rank
#                     receiver = op.dst.rank
#                     if sender != receiver:
#                         sop = instr_dag.add_send(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2+1, sendtb, ch)
#                         rop = instr_dag.add_recv(receiver, op.src, op.dst, op.steps_from_start*2+1, op.steps_to_end*2, recvtb, ch)
#                         sop.match = [rop]
#                     else:
#                         instr_dag.add_copy(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2, sendtb, ch)
#                 elif op.inst == ChunkInstruction.reduce:
#                     sender = op.src.rank
#                     receiver = op.dst.rank
#                     if sender != receiver:
#                         sop = instr_dag.add_send(sender, op.src, op.dst, op.steps_from_start*2,op.steps_to_end*2+1, sendtb, ch)
#                         rop = instr_dag.add_recv_reduce_copy(receiver, op.src, op.dst, op.steps_from_start*2+1, op.steps_to_end*2, recvtb, ch)
#                         sop.match = [rop]
#                     else:
#                         instr_dag.add_reduce(sender, op.src, op.dst, op.steps_from_start*2, op.steps_to_end*2, sendtb, ch)

#                 for o in op.next:
#                     heapq.heappush(frontier, ((o.steps_from_start, o.steps_to_end), o))
#                 visited.add(op)
#         instr_dag.convert_set_list() # Pre-emptively convert sets to lists
