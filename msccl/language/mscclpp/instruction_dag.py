# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from msccl.language.buffer import Buffer
from msccl.language.instruction_dag import (
    same_buf_dst,
    same_buf_src,
    same_src_dst_buffer_type,
)
from msccl.language.instruction_dag import InstructionDAG
from msccl.language.mscclpp.instruction_optimizer import InstructionOptimizer
from msccl.language.types import (
    Channel,
    ChannelType,
    ChunkRef,
    MscclppInstruction as Instruction,
    Op,
    ReplicationPolicy,
    Threadblock,
)


class MscclppInstructionDAG(InstructionDAG):
    def __init__(self, num_ranks, buffers):
        super().__init__(num_ranks, buffers)

    # InstructionDAG - adds a copy node
    def add_copy(self, rank, send_ref, recv_ref, tb, trans_from_packet=False, trans_to_packet=False):
        tb_step = self._get_tb_step(rank, tb)
        if trans_from_packet:
            op = Op(Instruction.copy_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step)
        elif trans_to_packet:
            op = Op(
                Instruction.transform_to_packet, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, step=tb_step
            )
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
    def add_put(self, rank, send_ref, recv_ref, tb, ch_type, use_packet=False):
        tb_step = self._get_tb_step(rank, tb)
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

    def add_flush(self, rank, send_ref, recv_ref, tb):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.flush,
            rank,
            send_ref,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ChannelType.proxy,
            step=tb_step,
        )
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
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

    def add_group_load_reduce(self, rank, send_refs, recv_ref, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.group_load_reduce,
            rank,
            None,
            recv_ref,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        for send_ref in send_refs:
            op.srcs.append((ChunkRef(send_ref.rank, send_ref.buffer, send_ref.index, send_ref.size), tb_step))
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op, read=True)

    def add_group_store(self, rank, send_ref, recv_refs, tb, ch_type):
        tb_step = self._get_tb_step(rank, tb)
        op = Op(
            Instruction.group_store,
            rank,
            send_ref,
            None,
            next=set(),
            prev=set(),
            tb=tb,
            channel_type=ch_type,
            step=tb_step,
        )
        for recv_ref in recv_refs:
            op.dsts.append((ChunkRef(recv_ref.rank, recv_ref.buffer, recv_ref.index, recv_ref.size), tb_step))
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        return op

    def complete_channels(self):
        send_op = [Instruction.put, Instruction.signal, Instruction.put_packet]
        recv_op = [Instruction.wait, Instruction.get, Instruction.read_reduce_copy]
        group_send_op = [Instruction.group_store]
        group_recv_op = [Instruction.group_load_reduce]
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                chans = set()
                for op in tb.ops:
                    if op.src != None:
                        src_buffer = (
                            Buffer.scratch
                            if op.src.buffer is not Buffer.input and op.src.buffer is not Buffer.output
                            else op.src.buffer
                        )
                    if op.dst != None:
                        dst_buffer = (
                            Buffer.scratch
                            if op.dst.buffer is not Buffer.input and op.dst.buffer is not Buffer.output
                            else op.dst.buffer
                        )
                    if op.channel_type == ChannelType.nvls:
                        if op.inst in group_send_op:
                            ranks = [dst[0].rank for dst in op.dsts]
                            ranks.append(op.rank)
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, ranks)
                            chans.add(chan)
                        elif op.inst in group_recv_op:
                            ranks = [src[0].rank for src in op.srcs]
                            ranks.append(op.rank)
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, ranks)
                            chans.add(chan)
                    else:
                        if op.inst in send_op:
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, op.dst.rank)
                            chans.add(chan)
                        elif op.inst in recv_op:
                            chan = Channel(src_buffer, dst_buffer, op.channel_type, op.src.rank)
                            chans.add(chan)
                tb.channels = list(chans)

    def remove_redundant_signal_wait(self):
        optimizer = InstructionOptimizer()
        # For packet ops, we can remove signal/wait
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst == Instruction.put_packet:
                        for next_op in op.next:
                            fused = optimizer.try_remove_op(next_op, next_op.inst == Instruction.signal)
                            if fused:
                                break
                    elif op.inst == Instruction.reduce_packet or op.inst == Instruction.copy_packet:
                        for prev_op in op.prev:
                            fused = optimizer.try_remove_op(prev_op, prev_op.inst == Instruction.wait)
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # put(src, sbuf, si, dst, dbuf, di) signal(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignal(src, sbuf, si, dst, dbuf, di)
    # put(src, sbuf, si, dst, dbuf, di) signal(src, sbuf, si, dst, dbuf, di) flush(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    def _fuse_instructions_using_proxy_channel(self):
        optimizer = InstructionOptimizer()
        inst_followup_map = {
            Instruction.put: Instruction.signal,
            Instruction.put_with_signal: Instruction.flush,
        }
        for rank, rank_tbs in enumerate(self.tbs):
            for tbid, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in inst_followup_map:
                        for next_op in op.next:
                            fused = optimizer.try_fuse_instructions_using_proxy_channel(
                                op, next_op, tb, queue, inst_followup_map[op.inst]
                            )
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) rrc(_,_,_,dst,dbuf,di) -> rrc(list[src,sbuf,si], dst, dbuf, di)
    # signal(_,_,_,dst,dbuf,di) signal(_,_,_,dst,dbuf,di) -> signal(_,_,_,list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    # reduce(_,_,_,dst,dbuf,di) reduce(_,_,_,dst,dbuf,di) -> reduce(list[src,sbuf,si], dst, dbuf, di)
    # reduce_packet(_,_,_,dst,dbuf,di) reduce_packet(_,_,_,dst,dbuf,di) -> reduce_packet(list[src,sbuf,si], dst, dbuf, di)
    def _fuse_same_instructions(self):
        optimizer = InstructionOptimizer()
        # Mapping instruction to their respective condition checks and same buffer function
        instruction_handlers = {
            Instruction.read_reduce_copy: same_buf_dst,
            Instruction.reduce: same_buf_dst,
            Instruction.reduce_packet: same_buf_dst,
            Instruction.signal: same_buf_src,
            Instruction.wait: same_buf_dst,
        }

        for _, rank_tbs in enumerate(self.tbs):
            for _, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    inst_type = op.inst
                    if inst_type in instruction_handlers:
                        for next_op in op.next:
                            same_buf_func = instruction_handlers[inst_type]
                            if optimizer.try_merge_same_instructions(op, next_op, tb, queue, inst_type, same_buf_func):
                                fused = True
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # rrc(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rrcs(_,_,_,_,_,_)
    # reduce(_,_,_,dst,dbuf,di) put(dst,dbuf,di,_,_,_) -> rs(_,_,_,_,_,_)
    def _optimize_rrcs_rs(self):
        optimizer = InstructionOptimizer()
        inst_types = [
            Instruction.read_reduce_copy,
            Instruction.reduce,
            Instruction.reduce_packet,
            Instruction.read_reduce_copy_send,
            Instruction.reduce_send,
            Instruction.reduce_send_packet,
        ]
        for _, rank_tbs in enumerate(self.tbs):
            for _, tb in rank_tbs.items():
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in inst_types:
                        for next_op in op.next:
                            fused = optimizer.try_fuse_with_put(op, next_op, tb, queue)
                            if fused:
                                break
                    if fused:
                        continue
                    queue = queue[1:]

    # merge ops which are independent of other operations and no other operations in between
    # get(src, sbuf. si, dst, dbuf, di) get(src, sbuf, si, dst, dbuf, di) -> get(list[src,sbuf,si], list[dst,dbuf,di])
    # put(src, sbuf, si, dst, dbuf, di) put(src, sbuf, si, dst, dbuf, di) -> put(list[src,sbuf,si], list[dst,dbuf,di])
    # putWithSignal/putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    # putWithSignal/putWithSignalAndFlush(src, sbuf, si, dst, dbuf, di)
    # -> putWithSignal/putWithSignalAndFlush(list[src,sbuf,si], list[dst,dbuf,di])
    # wait(src,sbuf,si,_,_,_) wait(src,sbuf,si,_,_,_) -> wait(list[src,sbuf,si],_,_,_,_])
    def _compact_instructions(self):
        optimizer = InstructionOptimizer()
        campactable_inst = [
            Instruction.get,
            Instruction.put,
            Instruction.put_packet,
            Instruction.put_with_signal,
            Instruction.put_with_signal_and_flush,
            Instruction.signal,
            Instruction.flush,
            Instruction.wait,
        ]
        for _, rank_tbs in enumerate(self.tbs):
            for _, tb in rank_tbs.items():
                if tb.id == -1:
                    continue
                queue = list(tb.ops)
                while len(queue) > 0:
                    op = queue[0]
                    fused = False
                    if op.inst in campactable_inst:
                        fused = optimizer.try_compact_instructions(op, tb, queue, op.inst, same_src_dst_buffer_type)

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
        self._fuse_instructions_using_proxy_channel()
        self._fuse_same_instructions()
        self._optimize_rrcs_rs()
        self._compact_instructions()

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
            elif replication_policy == ReplicationPolicy.interleaved:
                return index * instances + i * size
            return len(self.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

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
