# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language.instruction_dag import (
    merge_op,
    circular_dep_after_merge,
    same_chan_type,
    same_count,
)
from msccl.language.types import ChunkRef, MscclppInstruction as Instruction, Op, Threadblock


class InstructionOptimizer:
    def try_merge_same_instruction(
        self, op: Op, next_op: Op, tb: Threadblock, queue: list, inst_type: Instruction, same_buf_func: callable
    ) -> bool:
        """
        Attempts to merge two instruction if conditions are met.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param inst_type: The type of the instruction being processed.
        :param same_buf_func: The function to check if the buffer is the same (same_buf_dst or same_buf_src).
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == inst_type
            and same_buf_func(op, next_op)
            and same_count(op, next_op)
            and same_chan_type(op, next_op)
            and not circular_dep_after_merge(op, next_op)
        ):
            # Append the source chunks from next_op
            op.srcs.append(
                (
                    ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size),
                    next_op.step,
                )
            )
            # For 'signal' and 'wait' instructions, append destination chunks too
            if inst_type in [Instruction.signal, Instruction.wait, Instruction.flush]:
                op.dsts.append(
                    (
                        ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size),
                        next_op.step,
                    )
                )
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False

    def try_parallel_instruction(
        self, op: Op, tb: Threadblock, queue: list, inst_type: Instruction, same_src_dst_func: callable
    ) -> bool:
        """
        Try to parallelize the instructions which do not have dependencies.
        """
        if len(queue) > 1:
            next_op = queue[1]
            if (
                next_op.inst == inst_type
                and same_src_dst_func(op, next_op)
                and same_chan_type(op, next_op)
                and not circular_dep_after_merge(op, next_op)
            ):
                # Append the source and destination chunks from next_op
                op.dsts.append(
                    (
                        ChunkRef(next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size),
                        next_op.step,
                    )
                )
                op.srcs.append(
                    (
                        ChunkRef(next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size),
                        next_op.step,
                    )
                )
                merge_op(op, next_op)
                tb.ops.remove(next_op)
                queue.remove(next_op)
                return True
        return False
