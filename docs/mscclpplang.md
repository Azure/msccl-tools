## MSCCLPPLang Intruduction
MSCCLPPLang is a Python library for writing high-performance commnunication algorithms. It is designed to be easy to use and efficient, while providing a high-level interface for writing communication algorithms. MSCCLPPLang program will be compiled to json based execution plan, which can be executed by MSCCLPP runtime.

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
A chunk is a piece of data that is sent between nodes. It is the basic unit of data in MSCCLPPLang. Chunk can be a piece of data from input buffer, output buffer or intermediate buffer.
Example of creating a chunk:
```python
c = chunk(rank, Buffer.input, index, size)
```
- rank: the rank of the node that the chunk belongs to.
- Buffer: the buffer that the chunk belongs to. It can be Buffer.input, Buffer.output or Buffer.intermediate.
- index: the index of the chunk in the buffer.
- size: the size of the chunk.

Assume for input buffer, we split the input data into 4 chunks, then we can get the rank 0 chunks from 0 to 2 by:
```python
c = chunk(0, Buffer.input, 0, 2)
```

#### Operation
The operation can only be applied to the chunks. The operation can be a communication operation, such as put, get, reduce, etc.

## MSCCLPPLang APIs
## MSCCLPPLang Examples
