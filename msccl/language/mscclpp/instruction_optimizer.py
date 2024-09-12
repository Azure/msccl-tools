# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language.instruction_dag import (
    buf_dst_src_match,
    circular_dep_after_merge,
    merge_op,
    same_chan_type,
    same_count,
    same_buf_dst,
    same_buf_src,
)
from msccl.language.types import ChunkRef, ChannelType, MscclppInstruction as Instruction, Op, Threadblock


class InstructionOptimizer:

    def try_merge_same_instructions(
        self,
        op: Op,
        next_op: Op,
        tb: Threadblock,
        queue: list,
        expected_next_inst: Instruction,
        same_buf_func: callable,
    ) -> bool:
        """
        Attempts to merge two instruction if conditions are met.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param expected_next_inst: The instruction type expected for the next operation.
        :param same_buf_func: The function to check if the buffer is the same (same_buf_dst or same_buf_src).
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == expected_next_inst
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
            if expected_next_inst in [Instruction.signal, Instruction.wait, Instruction.flush]:
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

    def try_compact_instruction(
        self, op: Op, tb: Threadblock, queue: list, inst_type: Instruction, same_src_dst_func: callable
    ) -> bool:
        """
        Try to campact the instructions with the same instruction type. This optimization will
        compact multiple instructions of the same type into a single instruction.
        :param op: The current operation.
        :param seq_op: The sequential operation to merge with.
        :param tb: The task block containing the operations.
        :param queue: The queue of operations.
        :param inst_type: The type of the instruction being processed (get, put, put_packet).
        :return: True if operations are merged, False otherwise.
        """
        if len(queue) > 1:
            seq_op = queue[1]
            if (
                seq_op.inst == inst_type
                and same_src_dst_func(op, seq_op)
                and same_chan_type(op, seq_op)
                and same_count(op, seq_op)
                and not circular_dep_after_merge(op, seq_op)
            ):
                # Append the source and destination chunks from seq_op
                op.dsts.append(
                    (
                        ChunkRef(seq_op.dst.rank, seq_op.dst.buffer, seq_op.dst.index, seq_op.dst.size),
                        seq_op.step,
                    )
                )
                op.srcs.append(
                    (
                        ChunkRef(seq_op.src.rank, seq_op.src.buffer, seq_op.src.index, seq_op.src.size),
                        seq_op.step,
                    )
                )
                merge_op(op, seq_op)
                tb.ops.remove(seq_op)
                queue.remove(seq_op)
                return True
        return False

    def try_fuse_with_put(self, op: Op, next_op: Op, tb: Threadblock, queue: list) -> bool:
        """
        Attempts to fuse 'put' operations with other operations like read_reduce_copy, reduce, etc.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param inst_type: The type of the instruction being processed.
        :param chan_type: Channel type if applicable.
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == Instruction.put
            or next_op.inst == Instruction.put_packet
            and same_count(op, next_op)
            and buf_dst_src_match(op, next_op)
            and next_op.channel_type == ChannelType.sm
            and (op.channel_type == ChannelType.none or op.channel_type == ChannelType.sm)
            and not circular_dep_after_merge(op, next_op)
        ):
            if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                return False
            # Adjust instruction type and channel if needed
            if op.inst == Instruction.read_reduce_copy:
                op.inst = Instruction.read_reduce_copy_send
            elif op.inst == Instruction.reduce:
                op.inst = Instruction.reduce_send
                op.channel_type = ChannelType.sm
            elif op.inst == Instruction.reduce_packet:
                op.inst = Instruction.reduce_send_packet
                op.channel_type = ChannelType.sm
            # Append the destination chunk from next_op
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

    def try_fuse_instructions_using_proxy_channel(
        self, op: Op, next_op: Op, tb: Threadblock, queue: list, expected_next_inst: Instruction
    ) -> bool:
        """
        Attempts to fuse operations which using proxy channel.
        :param op: The current operation.
        :param next_op: The next operation to potentially merge with.
        :param tb: The thread block containing the operations.
        :param queue: The queue of operations.
        :param expected_next_inst: The instruction type expected for the next operation.
        :return: True if operations are merged, False otherwise.
        """
        if (
            next_op.inst == expected_next_inst
            and same_count(op, next_op)
            and same_buf_dst(op, next_op)
            and same_buf_src(op, next_op)
            and same_chan_type(op, next_op)
            and op.channel_type == ChannelType.proxy
            and not circular_dep_after_merge(op, next_op)
        ):
            if op.inst == Instruction.put and next_op.inst == Instruction.signal:
                op.inst = Instruction.put_with_signal
            elif op.inst == Instruction.put_with_signal and next_op.inst == Instruction.flush:
                op.inst = Instruction.put_with_signal_and_flush
            # Merge operations
            merge_op(op, next_op)
            tb.ops.remove(next_op)
            queue.remove(next_op)
            return True
        return False
