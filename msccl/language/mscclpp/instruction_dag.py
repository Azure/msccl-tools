# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from msccl.language.buffer import Buffer
from msccl.language.types import Channel, ChannelType
from msccl.language.instruction_dag import (
    buf_dst_src_match,
    merge_op,
    remove_op,
    circular_dep_after_merge,
    same_buf_dst,
    same_buf_src,
    same_chan_type,
    same_count,
    same_src_dst_buffer_type,
)
from msccl.language.instruction_dag import InstructionDAG
from msccl.language.types import ChunkRef, MscclppInstruction as Instruction, Op, ReplicationPolicy, Threadblock


class MscclppInstructionDAG(InstructionDAG):
    def __init__(self, num_ranks, buffers):
        super().__init__(num_ranks, buffers)

    # InstructionDAG - adds a copy node
    def add_copy(self, rank, send_ref, recv_ref, tb, use_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if use_packet:
            op = Op(Instruction.copy_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        else:
            op = Op(Instruction.copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        # Sending part of copy [Read]
        self._read(rank, srcbuffer, srcindex, size, op)
        # Receiving part of copy [Write]
        self._write(rank, dstbuffer, dstindex, size, op)
        return op

    # InstructionDAG - adds a redduce node
    def add_reduce(self, rank, send_ref, recv_ref, tb, use_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if use_packet:
            op = Op(Instruction.reduce_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        else:
            op = Op(Instruction.reduce, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        # Sending part of reduce
        self._read(rank, srcbuffer, srcindex, size, op)
        # Reduce part of copy
        self._write(rank, dstbuffer, dstindex, size, op, read=True)
        return op

    # InstructionDAG - adds a put node
    def add_put(self, rank, send_ref, recv_ref, tb, ch_type, use_packet=False, temp_chunk=None):
        tb_step = self._get_tb_step(rank, tb)
        if ch_type == ChannelType.proxy and temp_chunk is not None:
            op = Op(
                Instruction.transform_to_packet,
                rank,
                send_ref,
                temp_chunk,
                next=set(),
                prev=set(),
                tb=tb,
                channel_type=ch_type,
                step=tb_step,
            )
            tb_step = self._get_tb_step(rank, tb)
            op2 = Op(
                Instruction.put,
                rank,
                send_ref,
                recv_ref,
                next=set(),
                prev=set(),
                tb=tb,
                channel_type=ch_type,
                step=tb_step,
            )
            buffer = send_ref.buffer
            index = send_ref.index
            size = send_ref.size
            self._read(rank, buffer, index, size, op)
            self._write(rank, temp_chunk.buffer, temp_chunk.index, temp_chunk.size, op)
            self._read(rank, temp_chunk.buffer, temp_chunk.index, temp_chunk.size, op2)
            op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
            op2.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
            return op
        if use_packet:
            op = Op(
                Instruction.put_packet,
                rank,
                send_ref,
                recv_ref,
                next=set(),
                prev=set(),
                tb=tb,
                channel_type=ch_type,
                step=tb_step,
            )
        else:
            op = Op(
                Instruction.put,
                rank,
                send_ref,
                recv_ref,
                next=set(),
                prev=set(),
                tb=tb,
                channel_type=ch_type,
                step=tb_step,
            )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_get(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.get, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step
        )
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op)
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        return op

    # InstructionDAG - adds a signal node.
    def add_signal(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.signal,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        # treat signal as a write since it can not be executed parallelly with read operations
        self._write(rank, buffer, index, size, op)
        op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        return op

    def add_wait(self, rank, dst_ref, src_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.wait, rank, src_ref, dst_ref, next=set(), prev=set(), tb=tb, channel_type=ch_type, step=tb_step
        )
        buffer = dst_ref.buffer
        index = dst_ref.index
        size = dst_ref.size
        self._write(rank, buffer, index, size, op)
        op.srcs.append((ChunkRef(src_ref.rank, src_ref.buffer, src_ref.index, src_ref.size), tb_step))
        op.dsts.append((ChunkRef(dst_ref.rank, dst_ref.buffer, dst_ref.index, dst_ref.size), tb_step))
        return op

    def add_read_reduce(self, rank, send_ref, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.read_reduce_copy,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        self._write(rank, buffer, index, size, op, read=True)
        return op

    def complete_channels(self):
        send_op = [Instruction.put, Instruction.signal, Instruction.put_packet]
        recv_op = [Instruction.wait, Instruction.get, Instruction.read_reduce_copy]
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                chans = set()
                for op in tb.ops:
                    src_buffer = (
                        Buffer.scratch
                        if op.src.buffer is not Buffer.input and op.src.buffer is not Buffer.output
                        else op.src.buffer
                    )
                    dst_buffer = (
                        Buffer.scratch
                        if op.dst.buffer is not Buffer.input and op.dst.buffer is not Buffer.output
                        else op.dst.buffer
                    )
                    if op.inst in send_op:
                        chan = Channel(src_buffer, dst_buffer, op.channel_type, op.dst.rank)
                        chans.add(chan)
                    elif op.inst in recv_op:
                        chan = Channel(src_buffer, dst_buffer, op.channel_type, op.src.rank)
                        chans.add(chan)
                tb.channels = list(chans)

    def _optimize_redundant_signal_wait(self):
        # For packet ops, we can remove signal/wait
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.put_packet:
                        fused = False
                        for next_op in op.next:
                            if next_op.inst == Instruction.signal:
                                remove_op(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.reduce_packet or op.inst == Instruction.copy_packet:
                        fused = False
                        for prev_op in op.prev:
                            if prev_op.inst == Instruction.wait:
                                remove_op(prev_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) rrc(_,_,_,dst,dbuf,di) -> rrc(list[src,sbuf,si], dst, dbuf, di)
    # signal(_,_,_,dst,dbuf,di) signal(_,_,_,dst,dbuf,di) -> signal(_,_,_,list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    # reduce(_,_,_,dst,dbuf,di) reduce(_,_,_,dst,dbuf,di) -> reduce(list[src,sbuf,si], dst, dbuf, di)
    # reduce_packet(_,_,_,dst,dbuf,di) reduce_packet(_,_,_,dst,dbuf,di) -> reduce_packet(list[src,sbuf,si], dst, dbuf, di)
    def _optimize_rrc_r_signal_wait(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.read_reduce_copy:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.read_reduce_copy
                                and same_count(op, next_op)
                                and same_buf_dst(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                op.srcs.append(
                                    (
                                        ChunkRef(
                                            next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.reduce:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.reduce
                                and same_buf_dst(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                op.srcs.append(
                                    (
                                        ChunkRef(
                                            next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.reduce_packet:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.reduce_packet
                                and same_buf_dst(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                op.srcs.append(
                                    (
                                        ChunkRef(
                                            next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.signal:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.signal
                                and same_buf_src(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                op.dsts.append(
                                    (
                                        ChunkRef(
                                            next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                op.srcs.append(
                                    (
                                        ChunkRef(
                                            next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    elif op.inst == Instruction.wait:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.wait
                                and same_buf_dst(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                op.srcs.append(
                                    (
                                        ChunkRef(
                                            next_op.src.rank, next_op.src.buffer, next_op.src.index, next_op.src.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                op.dsts.append(
                                    (
                                        ChunkRef(
                                            next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rrcs(_,_,_,_,_,_)
    # reduce(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rs(_,_,_,_,_,_)
    def _optimize_rrcs_rs(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.read_reduce_copy or op.inst == Instruction.read_reduce_copy_send:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.put
                                and same_count(op, next_op)
                                and buf_dst_src_match(op, next_op)
                                and same_chan_type(op, next_op)
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                                    continue
                                if op.inst == Instruction.read_reduce_copy:
                                    op.inst = Instruction.read_reduce_copy_send
                                op.dsts.append(
                                    (
                                        ChunkRef(
                                            next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    if op.inst == Instruction.reduce or op.inst == Instruction.reduce_send:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.put
                                and same_count(op, next_op)
                                and buf_dst_src_match(op, next_op)
                                and next_op.channel_type == ChannelType.sm
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                                    continue
                                if op.inst == Instruction.reduce:
                                    op.inst = Instruction.reduce_send
                                    op.channel_type = ChannelType.sm
                                op.dsts.append(
                                    (
                                        ChunkRef(
                                            next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    if op.inst == Instruction.reduce_packet or op.inst == Instruction.reduce_send_packet:
                        fused = False
                        for next_op in op.next:
                            if (
                                next_op.inst == Instruction.put_packet
                                and same_count(op, next_op)
                                and buf_dst_src_match(op, next_op)
                                and next_op.channel_type == ChannelType.sm
                                and not circular_dep_after_merge(op, next_op)
                            ):
                                if len(op.dsts) > 0 and op.dsts[0][0].buffer != next_op.dst.buffer:
                                    continue
                                if op.inst == Instruction.reduce_packet:
                                    op.inst = Instruction.reduce_send_packet
                                    op.channel_type = ChannelType.sm
                                op.dsts.append(
                                    (
                                        ChunkRef(
                                            next_op.dst.rank, next_op.dst.buffer, next_op.dst.index, next_op.dst.size
                                        ),
                                        next_op.step,
                                    )
                                )
                                merge_op(op, next_op)
                                tb.ops.remove(next_op)
                                queue.remove(next_op)
                                fused = True
                                break
                        if fused:
                            continue
                    queue = queue[1:]

    # get(src, sbuf. si, dst, dbuf, di) get(src, sbuf, si, dst, dbuf, di) -> get(list[src,sbuf,si], list[dst,dbuf,di])
    # put(src, sbuf, si, dst, dbuf, di) put(src, sbuf, si, dst, dbuf, di) -> put(list[src,sbuf,si], list[dst,dbuf,di])
    def _optimize_get_put(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.get:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if (
                                seq_op.inst == Instruction.get
                                and same_src_dst_buffer_type(op, seq_op)
                                and same_chan_type(op, seq_op)
                                and same_count(op, seq_op)
                                and not circular_dep_after_merge(op, seq_op)
                            ):
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
                                fused = True
                        if fused:
                            continue
                    elif op.inst == Instruction.put:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if (
                                seq_op.inst == Instruction.put
                                and same_src_dst_buffer_type(op, seq_op)
                                and same_chan_type(op, seq_op)
                                and same_count(op, seq_op)
                                and not circular_dep_after_merge(op, seq_op)
                            ):
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
                                fused = True
                        if fused:
                            continue
                    elif op.inst == Instruction.put_packet:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if (
                                seq_op.inst == Instruction.put_packet
                                and same_src_dst_buffer_type(op, seq_op)
                                and same_chan_type(op, seq_op)
                                and same_count(op, seq_op)
                                and not circular_dep_after_merge(op, seq_op)
                            ):
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
                                fused = True
                        if fused:
                            continue
                    queue = queue[1:]

    # For signal/wait ops, if they are independent of other operations and no other operations in between,
    # then merge them into a single signal/wait op
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    def _parallel_signal_wait(self):
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                if tbid == -1:
                    continue
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    if op.inst == Instruction.signal:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if (
                                seq_op.inst == Instruction.signal
                                and same_src_dst_buffer_type(op, seq_op)
                                and same_chan_type(op, seq_op)
                                and not circular_dep_after_merge(op, seq_op)
                            ):
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
                                fused = True
                        if fused:
                            continue
                    elif op.inst == Instruction.wait:
                        fused = False
                        if len(queue) > 1:
                            seq_op = queue[1]
                            if (
                                seq_op.inst == Instruction.wait
                                and same_src_dst_buffer_type(op, seq_op)
                                and same_chan_type(op, seq_op)
                                and not circular_dep_after_merge(op, seq_op)
                            ):
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
                                fused = True
                        if fused:
                            continue
                    queue = queue[1:]

    def _get_tb_step(self, rank: int, tb: int):
        if tb in self.tb_steps[rank]:
            self.tb_steps[rank][tb] += 1
            return self.tb_steps[rank][tb]
        else:
            self.tb_steps[rank][tb] = 0
            return 0

    def optimize(self):
        self._optimize_redundant_signal_wait()
        self._optimize_rrc_r_signal_wait()
        self._optimize_rrcs_rs()
        self._optimize_get_put()

        self._parallel_signal_wait()

    def replicate(self, instances: int, replication_policy: ReplicationPolicy):
        # update op step
        for rank, rank_tbs in enumerate(self.tbs):
            for _, tb in rank_tbs.items():
                for id, op in enumerate(tb.ops):
                    op.step = id

        if instances == 1:
            self.instanced_tbs = self.tbs
            return

        self.instanced_tbs = []
        for _ in range(self.num_ranks):
            self.instanced_tbs.append({})

        def is_scratch(buffer):
            return buffer != Buffer.input and buffer != Buffer.output

        def get_new_index(rank, buffer, index, size, i):
            # Scratch buffers always use batched
            if is_scratch(buffer):
                buf_instance_len = self.buffers[rank][buffer].instance_size()
                return buf_instance_len * i + index
            return len(self.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        if replication_policy == ReplicationPolicy.duplicated:
            for i in range(instances):
                # Generate all the threadblocks and ops
                for rank, rank_tbs in enumerate(self.tbs):
                    # rank_channels = self.num_channels[rank]
                    for tbid, tb in rank_tbs.items():
                        itbid = tbid * instances + i
                        itb = Threadblock(id=itbid)
                        itb.ops = [None] * len(tb.ops)
                        for s, op in enumerate(tb.ops):
                            isrc = get_instance_ref(op.src)
                            idst = get_instance_ref(op.dst)
                            idepends = []
                            # Note: We don't need the fill out the rest of the metadata since replication is the last optimization
                            iop = Op(
                                op.inst, op.rank, isrc, idst, idepends, op.step, itbid, channel_type=op.channel_type
                            )
                            itb.ops[s] = iop
                            for src, step in op.srcs:
                                isrc = get_instance_ref(src)
                                iop.srcs.append((isrc, step))
                            for dst, step in op.dsts:
                                idst = get_instance_ref(dst)
                                iop.dsts.append((idst, step))
                        for chan in tb.channels:
                            itb.channels.append(chan)
                        self.instanced_tbs[op.rank][itbid] = itb

            # Redo dependency analysis
            for rank, rank_tbs in enumerate(self.tbs):
                for tbid, tb in rank_tbs.items():
                    for i in range(instances):
                        itbid = tbid * instances + i
                        itb = self.instanced_tbs[rank][itbid]
                        for op, iop in zip(tb.ops, itb.ops):
                            iop.depends = [None] * len(op.depends)
                            for s, dep in enumerate(op.depends):
                                dep_tbid = dep.tb
                                dep_itbid = dep_tbid * instances + i
                                dep_step = dep.step
                                iop.depends[s] = self.instanced_tbs[op.rank][dep_itbid].ops[dep_step]
