from msccl.language.collectives import AllReduce

def allreduce_ring():
    for r in range(0, size):
        send_to_peer = (r + 1) % size
        recv_from_peer = (r - 1) % size
        send_ch = get_channel(src_rank, src_buffer, send_to_peer, dst_buffer, channel_type, tag)
        recv_ch = get_channel(src_rank, src_buffer, recv_from_peer, dst_buffer, channel_type, tag)
        # for channel the key is (src_rank, src_buffer, dst_rank, dst_buffer, channel_type)
        for i in range(0, size - 1):
            send_index = (r + i) % size
            c = chunk(r, buffer_type, send_index, size)
            c.put(send_to_peer, dst_buffer, send_index, sendtbs=[tb_list], ch=ch)
            # need barrier to make sure the put is done
            rank = get_rank(r)
            rank.barrier(tbs=[tb_list])
            send_ch.signal(tb=sendtb)
            recv_ch.wait(tb=recvtb)
