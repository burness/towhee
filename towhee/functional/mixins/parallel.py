# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
from queue import Queue
import asyncio

from towhee.functional.option import Option
from towhee.hparam.hyperparameter import param_scope


class ParallelMixin:
    """
    Mixin for parallel execution.

    Examples:
    >>> from towhee.functional import DataCollection
    >>> def add_1(x):
    ...     return x+1
    >>> result = DataCollection.range(1000).map(add_1).parallel(5).to_list()
    >>> len(result)
    1000
    >>> result = DataCollection.range(1000).pmap(add_1, 10).pmap(add_1, 10).to_list()
    >>> result[990:]
    [992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001]
    """

    def __init__(self) -> None:
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_executor') and isinstance(
                parent._executor, concurrent.futures.ThreadPoolExecutor):
            self.set_parallel(executor=parent._executor)

    def set_parallel(self, num_worker=None, executor=None):
        """
        set parallel execution

        Examples:
        >>> from towhee.functional import DataCollection
        >>> import threading
        >>> stage_1_thread_set = set()
        >>> stage_2_thread_set = set()
        >>> result = (
        ...     DataCollection.range(1000).stream().set_parallel(4)
        ...     .map(lambda x: stage_1_thread_set.add(threading.current_thread().ident))
        ...     .map(lambda x: stage_2_thread_set.add(threading.current_thread().ident)).to_list()
        ... )
        >>> len(stage_2_thread_set)
        3
        """
        if executor is not None:
            self._executor = executor
        if num_worker is not None:
            self._executor = concurrent.futures.ThreadPoolExecutor(num_worker)
        return self

    def parallel(self, num_worker):
        executor = concurrent.futures.ThreadPoolExecutor(num_worker)
        queue = Queue(maxsize=num_worker)
        gen = iter(self)
        cnt = num_worker

        def worker():
            nonlocal cnt
            for x in gen:
                queue.put(x)
            cnt -= 1

        for _ in range(num_worker):
            executor.submit(worker)

        def inner():
            while cnt > 0 or not queue.empty():
                yield queue.get()
            executor.shutdown()

        return self.factory(inner())

    def pmap(self, unary_op, num_worker=None, executor=None):
        """
        apply `unary_op` with parallel execution

        Examples:
        >>> from towhee.functional import DataCollection
        >>> import threading
        >>> stage_1_thread_set = set()
        >>> stage_2_thread_set = set()
        >>> result = (
        ...     DataCollection.range(1000).stream()
        ...     .pmap(lambda x: stage_1_thread_set.add(threading.current_thread().ident), 5)
        ...     .pmap(lambda x: stage_2_thread_set.add(threading.current_thread().ident), 4).to_list()
        ... )
        >>> len(stage_1_thread_set)
        4
        >>> len(stage_2_thread_set)
        3
        """
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(num_worker)
        num_worker = executor._max_workers  # pylint: disable=protected-access
        queue = Queue(maxsize=num_worker)
        loop = asyncio.new_event_loop()
        flag = True

        def make_task(x):

            def task_wrapper():
                if isinstance(x, Option):
                    return x.map(unary_op)
                else:
                    return unary_op(x)

            return task_wrapper

        async def worker():
            buff = []
            for x in self:
                if len(buff) == num_worker:
                    queue.put(await buff.pop(0))
                buff.append(loop.run_in_executor(executor, make_task(x)))
            while len(buff) > 0:
                queue.put(await buff.pop(0))
            nonlocal flag
            flag = False

        def worker_wrapper():
            loop.run_until_complete(worker())

        executor.submit(worker_wrapper)

        def inner():
            nonlocal flag
            while flag or not queue.empty():
                yield queue.get()
            # executor.shutdown()

        return self.factory(inner())


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest
    doctest.testmod(verbose=False)