import time
import os
from typing import Optional, Iterator, Callable
from contextlib import contextmanager

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # 可选依赖：无则降级

@contextmanager
def measure(name: str = "block", realtime: bool = False, interval: float = 1.0) -> Iterator[None]:
    """
    通用测量：时间、CPU、内存。
    - name: 标记名，会在打印中显示
    - realtime: 是否实时采样（周期采样 CPU%、RSS）
    - interval: 采样间隔秒（realtime=True 时生效）
    使用：
        with measure("build_embeddings", realtime=True, interval=1.0):
            snapshot_embeddings = build_embeddings(handler)
    打印：
        [MEASURE] name | wall=..s cpu%=.. mem_avg=..MB mem_max=..MB
    """
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    samples_cpu = []
    samples_mem = []
    stop = False

    def sampler():
        if psutil is None:
            return
        proc = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)  # 预热，避免第一下 0.0
        while not stop:
            cpu_pct = float(psutil.cpu_percent(interval=interval))
            rss_mb = float(proc.memory_info().rss) / (1024 * 1024)
            samples_cpu.append(cpu_pct)
            samples_mem.append(rss_mb)

    import threading
    th: Optional[threading.Thread] = None
    if realtime and psutil is not None:
        th = threading.Thread(target=sampler, daemon=True)
        th.start()

    try:
        yield
    finally:
        wall = time.perf_counter() - t0_wall
        cpu_time = time.process_time() - t0_cpu
        stop = True
        if th is not None:
            th.join(timeout=max(1.0, interval))

        # 统计
        if samples_cpu:
            cpu_avg = sum(samples_cpu) / len(samples_cpu)
        else:
            cpu_avg = (cpu_time / wall * 100.0) if wall > 0 else None

        mem_avg = sum(samples_mem) / len(samples_mem) if samples_mem else None
        mem_max = max(samples_mem) if samples_mem else None

        # 若未实时采样，尝试给一次当前 RSS
        if psutil is not None and mem_avg is None:
            rss = psutil.Process(os.getpid()).memory_info().rss
            mem_avg = float(rss) / (1024 * 1024)
            mem_max = mem_avg

        # 打印
        wall_s = f"{wall:.3f}" if wall is not None else "UNKNOWN"
        cpu_s = f"{cpu_avg:.1f}" if cpu_avg is not None else "UNKNOWN"
        mem_avg_s = f"{mem_avg:.1f}" if mem_avg is not None else "UNKNOWN"
        mem_max_s = f"{mem_max:.1f}" if mem_max is not None else "UNKNOWN"
        print(f"[MEASURE] {name} | wall={wall_s}s cpu%={cpu_s} mem_avg={mem_avg_s}MB mem_max={mem_max_s}MB")


def measure_func(name: Optional[str] = None, realtime: bool = False, interval: float = 1.0) -> Callable:
    """装饰器版：@measure_func("train", realtime=True)
    打印同 measure 上下文。
    """
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            lbl = name or fn.__name__
            with measure(lbl, realtime=realtime, interval=interval):
                return fn(*args, **kwargs)
        return wrapper
    return deco
