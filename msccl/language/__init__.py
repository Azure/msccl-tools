# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language.chunk import *
from msccl.language.buffer import *
from msccl.language.instruction_dag import *
import msccl.language.mscclpp as mscclpp
from msccl.language.mscclpp import *
from typing import Union

_current_program = None


def _curr():
    global _current_program
    if _current_program == None and mscclpp._current_program == None:
        raise RuntimeError("No Program in context")
    if _current_program == None:
        return mscclpp._current_program
    return _current_program


def chunk(rank, buffer, index, size=1) -> Union[mscclpp.Ref, Ref]:
    if _curr().buffers[rank][buffer][index] is None:
        return None
    return _curr().get_ref(rank, buffer, index, size)

def rank(rank) -> mscclpp.RankRef:
    return _curr().get_rank_ref(rank)


def create_scratch(rank, name):
    return _curr().create_scratch(rank, name)


def Check():
    return _curr().check()
