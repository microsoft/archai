# Copyright (c) EleutherAI.
# Licensed under the MIT license.
# https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py

from __future__ import annotations

from typing import Any, Optional

REQUEST_RETURN_LENGTHS = {
    "generate": None,
    "greedy_until": None,
    "loglikelihood": 2,
    "loglikelihood_rolling": None,
}


class Request:
    def __init__(self, request_type: str, args: Any, index: Optional[int] = None) -> None:
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError("The request type {} is not implemented!".format(request_type))

        self.request_type = request_type
        self.args = args
        self.index = index

    def __iter__(self) -> Request:
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, i)

    def __getitem__(self, i: int) -> Request:
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i)

    def __eq__(self, other: Request) -> bool:
        return self.request_type == other.request_type and self.args == other.args and self.index == other.index

    def __repr__(self) -> str:
        return f"Req_{self.request_type}{self.args}[{self.index}]\n"


class RequestFactory:
    def __getattr__(self, attr: str) -> Request:
        def fn(*args):
            return Request(attr, args)

        return fn


rf = RequestFactory()
