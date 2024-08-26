## MSCCLPPLang Intruduction
MSCCLPPLang is a Python library for writing high-performance commnunication algorithms. It is designed to be easy to use and efficient, while providing a high-level interface for writing communication algorithms. MSCCLPPLang program will be compiled to json based execution plan, which can be executed by MSCCLPP executor.

## How to Install MSCCLPPLang
```bash
git clone https://github.com/microsoft/msccl-tools.git
cd msccl-tools
pip install .
```

## How MSCCLPPLang Works
MSCCLPPLang provides a high-level interface for writing communication algorithms. We treat the communication algorithm as a graph, where the nodes are the data and the edges are the communication operations. The graph is represented as a Python program, which is compiled to a json based execution plan.

### Core concepts

#### Chunk
A chunk is a piece of data that is sent between GPUs. It is the basic unit of data in MSCCLPPLang. Chunk can be a piece of data from input buffer, output buffer or intermediate buffer.
Example of creating a chunk:
```python
c = chunk(rank, Buffer.input, index, size)
```
- rank: the rank of the GPU that the chunk belongs to.
- buffer: the buffer that the chunk belongs to. It can be Buffer.input, Buffer.output or Buffer.intermediate.
- index: the index of the chunk in the buffer.
- size: the size of the chunk.

Assume we split the input data in the buffer into 4 chunks. On GPU rank 0, we can retrieve the chunks from indices 0 to 2 using the following command:
```python
c = chunk(0, Buffer.input, 0, 2)
```

#### Operation
The operation can only be applied to the chunks. We provide a set of communicatoin operations for the users to use. For example, the `put` operation is used to send the data from one GPU to another GPU. The `get` operation is used to receive the data from another GPU.

***Please notice***: MSCCLPPLang only provides one-sided communication operations. The user needs to make sure that the data is ready to be sent or received before calling the communication operations. Also we provides `wait/signal` operations to synchronize the communication across GPUs.

#### Channel
A channel is a communication channel between two GPUs. It is used to send and receive data between GPUs. We supports two types of channel: `ChannelType.sm` and `ChannelType.proxy`.

`ChannelType.sm` is used for communication between GPUs on the same node. This channel will using GPU processors to transfer data.

`ChannelType.proxy` is used for communication between GPUs, whether they are on different nodes or the same node. This channel will offload the data transfer to CPU processors, which can provide better throughput compared to `ChannelType.sm`. However, this comes at the cost of higher latency compared to `ChannelType.sm`.

#### Thread Block

We can assign operations to a thread block. The thread block is a group of threads that are executed together on the GPU. In the operation function, we can specify the thread block that the operation belongs to via `sendtb` or `recvtb` parameter.

#### Kernel fusion
MSCCLPPLang provides a kernel fusion mechanism to fuse multiple operations into a single kernel. This can reduce the overhead of launching multiple kernels. When user create the MSCCLPPLang program, it can specify the `instr_fusion` parameter to enable the kernel fusion. By default, the kernel fusion is enabled.

## MSCCLPPLang APIs

### Basic APIs
- `chunk(rank, buffer, index, size)`: create a chunk.
- `put(self, dst, chunk, index, sendtb, chan_type)`: send the data from one GPU to another GPU. User can specify the index of the chunk in the destination buffer, the sendtb and the channel type.
- `get(self, src, chunk, index, recvtb, chan_type)`: receive the data from another GPU. User can specify the index of the chunk in the destination buffer, the recvtb and the channel type.
- `signal(self, dst, buffer, index, sendtb, chan_type)`: send a signal to another GPU.
- `wait(self, src, buffer, index, recvtb, chan_type)`: wait for a signal from another GPU.
- `copy(self, dst, buffer, index, sendtb)`: copy the data from one buffer to another buffer in the same GPU.
- `reduce(self, other_chunkref, recvtb, channel_type)`: Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref

### Packet APIs
Packet APIs are used when user wants to use LL algorithm. The packet APIs are similar to the basic APIs, it will packet the data and flags into a packet and send the packet to the destination GPU. The destination GPU will unpack the packet and get the data and flags. So no synchronization is needed when using packet APIs.
- `packet_put(self, dst, chunk, index, sendtb, chan_type)`: send the data from one GPU to another GPU using packet.
- `copy_packet(self, dst, buffer, index, sendtb)`: copy the data from one buffer to another buffer in the same GPU using packet.
- `reduce_packet(self, other_chunkref, recvtb)`: Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref using packet.
