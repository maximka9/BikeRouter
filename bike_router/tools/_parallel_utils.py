"""Общие утилиты для авто-параллелизма экспериментов (без новых CLI-флагов)."""

from __future__ import annotations

import os
from typing import Any, Iterable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def auto_cpu_count() -> int:
    """Число логических процессоров для планирования пулов."""
    try:
        n = os.process_cpu_count()
    except AttributeError:
        n = None  # type: ignore[assignment]
    if n is not None and int(n) > 0:
        return int(n)
    c = os.cpu_count()
    return int(c) if c and c > 0 else 1


def auto_worker_count(task_count: int, *, memory_heavy: bool = False) -> int:
    """Число воркеров ProcessPool: не больше задач и не больше политики по CPU/RAM.

    * ``memory_heavy=True`` (route batch): ориентир 4–6 процессов, не выше ``cpu_count-1``.
    * ``memory_heavy=False`` (тяжёлые тайлы / кандидаты моделей): до ``cpu_count - 1``.
    * При ``task_count <= 1`` — последовательно (1).
    """
    tc = max(0, int(task_count))
    if tc <= 1:
        return 1
    c = auto_cpu_count()
    cap_cpu = max(1, c - 1)
    if memory_heavy:
        preferred = min(6, max(4, cap_cpu))
        return max(1, min(cap_cpu, preferred, tc))
    return max(1, min(cap_cpu, tc))


def chunked(items: Sequence[T], chunk_size: int) -> Iterator[List[T]]:
    """Непересекающиеся чанки фиксированного размера (последний может быть короче)."""
    sz = max(1, int(chunk_size))
    seq = list(items)
    for i in range(0, len(seq), sz):
        yield seq[i : i + sz]


def chunked_iterable(items: Iterable[T], chunk_size: int) -> Iterator[List[T]]:
    """Чанки из произвольного итерируемого (материализует элементы по мере чтения)."""
    sz = max(1, int(chunk_size))
    buf: List[T] = []
    for x in items:
        buf.append(x)
        if len(buf) >= sz:
            yield buf
            buf = []
    if buf:
        yield buf
