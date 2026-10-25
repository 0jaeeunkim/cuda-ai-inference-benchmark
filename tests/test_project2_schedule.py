"""Exercise real enqueue logic on an adversarial stream DAG, without CUDA.

This detects slot overwrite races; it is NOT GPU/model correctness evidence.
"""

import contextlib
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project2_pipeline import Prepared
import project2_protocol as p


class Engine:
    def __init__(self, seed, ignore_reuse_wait=False):
        self.nodes, self.done = [], set()
        self.current = None
        self.random = random.Random(seed)
        self.ignore_reuse_wait = ignore_reuse_wait
        self.cuda = SimpleNamespace(stream=self.stream)

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def enqueue(self, action):
        stream = self.current
        dependencies = set(stream.waits)
        if stream.tail is not None:
            dependencies.add(stream.tail)
        index = len(self.nodes)
        self.nodes.append((dependencies, action))
        stream.tail = index
        return index

    def drain(self, until=None):
        while len(self.done) < len(self.nodes) and (until is None or until not in self.done):
            ready = [i for i, (dependencies, _) in enumerate(self.nodes)
                     if i not in self.done and dependencies <= self.done]
            if not ready:
                raise AssertionError("Stream/event dependency cycle")
            selected = self.random.choice(ready)
            self.nodes[selected][1]()
            self.done.add(selected)


class Stream:
    def __init__(self, engine, name):
        self.engine, self.name = engine, name
        self.tail, self.waits = None, set()

    def wait_event(self, event):
        if self.name == "copy" and self.engine.ignore_reuse_wait:
            return
        if event.node is None:
            raise AssertionError("Wait submitted before producer event record")
        self.waits.add(event.node)


class Event:
    node = None

    def record(self, stream):
        self.node = stream.tail


class Tensor:
    def __init__(self, engine, storage, start, count, width):
        self.engine, self.storage = engine, storage
        self.start, self.count, self.shape = start, count, (count, width)

    def values(self):
        return self.storage[self.start:self.start + self.count]

    def store(self, values):
        if len(values) != self.count:
            raise AssertionError("Full/tail output shape changed")
        self.storage[self.start:self.start + self.count] = values

    def copy_(self, source, non_blocking):
        node = self.engine.enqueue(lambda: self.store(source.values()))
        if not non_blocking:
            self.engine.drain(until=node)

    def data_ptr(self):
        return id(self.storage) + self.start * 8


def exercise(variant, batch, count, seed, ignore_reuse_wait=False):
    engine = Engine(seed, ignore_reuse_wait)
    compute = Stream(engine, "compute")
    copy = Stream(engine, "copy") if variant == "D" else compute
    source, output = list(range(count)), [None] * count
    slots = [[None] * batch for _ in range(2)]

    def model(**inputs):
        src = inputs["input_ids"]
        logits = Tensor(engine, [None] * src.count, 0, src.count, 2)
        engine.enqueue(lambda: logits.store([(x, -x) for x in src.values()]))
        return SimpleNamespace(logits=logits)

    runner = Prepared.__new__(Prepared)
    runner.torch, runner.model, runner.variant = engine, model, variant
    runner.compute, runner.copy, runner.inputs = compute, copy, {"input_ids": None}
    runner.plan = []
    for i, (a, b) in enumerate(p.spans(batch, count)):
        runner.plan.append({"start": a, "end": b, "slot": i % 2,
                            "host": {"input_ids": Tensor(engine, source, a, b - a, 64)},
                            "device": {"input_ids": Tensor(engine, slots[i % 2], 0, b - a, 64)},
                            "output": Tensor(engine, output, a, b - a, 2),
                            "ready": Event(), "consumed": Event()})
    records = runner.enqueue(inspect=True)
    engine.drain()
    return output, records


class SchedulingTests(unittest.TestCase):
    def test_real_enqueue_preserves_rows_under_adversarial_interleavings(self):
        for variant in "ABCD":
            for batch, count in p.SMOKE_CASES:
                for seed in range(12):
                    with self.subTest(variant=variant, batch=batch, seed=seed):
                        output, records = exercise(variant, batch, count, seed)
                        self.assertEqual(output, [(i, -i) for i in range(count)])
                        self.assertEqual(len(records), len(p.spans(batch, count)))

    def test_simulator_detects_missing_slot_reuse_dependency(self):
        corrupted = 0
        for seed in range(12):
            output, _ = exercise("D", 8, 19, seed, ignore_reuse_wait=True)
            corrupted += output != [(i, -i) for i in range(19)]
        self.assertGreater(corrupted, 0)


if __name__ == "__main__":
    unittest.main()
