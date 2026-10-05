"""
Event-driven engine for layer-streaming inference over a host-mediated CXL path.

Resources
---------
* one copy engine / link: serves one transfer at a time (non-preemptive, as a CUDA copy
  stream does); demand fetches are served before queued prefetches,
* one compute stream: layers execute strictly in trace order,
* a weight cache in HBM of ``cache_bytes`` bytes holding resident *and in-flight* units;
  pinned units are loaded before measurement and never evicted.

Stochastic effects (disabled with ``deterministic=True``)
---------------------------------------------------------
* lognormal noise on every copy and every layer execution,
* a slow AR(1) drift of the link bandwidth shared by successive copies.

The engine itself contains no policy: prefetching, victim selection and pinning are
supplied by ``policies.py``.
"""
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hw import GPU, Link
from .models import Trace

INF = float("inf")


@dataclass
class Job:
    uid: int
    nbytes: float
    prio: int            # 0 demand, 1 prefetch
    seq: int
    gather: bool = False
    t_issue: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    done: bool = False


@dataclass
class Result:
    total_time: float
    iter_times: List[float]
    stall_time: float
    compute_time: float
    link_busy: float
    bytes_fetched: float
    demand_fetches: int
    late_prefetches: int
    hits: int
    accesses: int
    prefetch_issued: int
    prefetch_wasted: int
    iter_stall: List[float] = field(default_factory=list)

    def steady(self, skip: int = 1) -> float:
        xs = self.iter_times[skip:] if len(self.iter_times) > skip else self.iter_times
        return sum(xs) / len(xs)

    def steady_stall(self, skip: int = 1) -> float:
        xs = self.iter_stall[skip:] if len(self.iter_stall) > skip else self.iter_stall
        return sum(xs) / len(xs)


class Engine:
    def __init__(self, trace: Trace, gpu: GPU, link: Link, cache_bytes: float, policy,
                 seed: int = 0, deterministic: bool = False, compute_sigma: float = 0.03):
        self.trace = trace
        self.gpu = gpu
        self.link = link
        self.cap = float(cache_bytes)
        self.policy = policy
        self.rng = random.Random(seed)
        self.det = deterministic
        self.compute_sigma = compute_sigma
        self.units = trace.units
        # cache state
        self.state: Dict[int, int] = {u: 0 for u in self.units}   # 0 absent, 1 in flight, 2 resident
        self.ready: Dict[int, float] = {}
        self.last_use: Dict[int, float] = {u: -INF for u in self.units}
        self.freq: Dict[int, int] = {u: 0 for u in self.units}
        self.ins_seq: Dict[int, int] = {u: 0 for u in self.units}
        self.used = 0.0
        self.pinned = set()
        self.protect = set()
        self.demand_blocked = False   # a demand fetch is waiting for room: prefetches must not take it
        self._seq = 0
        self.useful_pf: Dict[int, bool] = {}
        # link state
        self.pending: List[Job] = []
        self.cur: Optional[Job] = None
        self.link_busy = 0.0
        self._drift_grid: List[float] = []
        self._drift_rng = random.Random(seed * 7919 + 17)
        # progress (visible to policies)
        self.i = 0                    # index of access being waited for / executed
        self.now = 0.0
        self.cur_start = 0.0          # compute start of access i (or None before it starts)
        self.computing = False
        # stats
        self.bytes_fetched = 0.0
        self.demand_fetches = 0
        self.late_prefetches = 0
        self.hits = 0
        self.prefetch_issued = 0
        self.prefetch_wasted = 0
        self.stall = 0.0
        self.comp = 0.0

    # ------------------------------------------------------------------ link
    def _noise(self, sigma):
        return 1.0 if self.det or sigma == 0 else self.rng.lognormvariate(0.0, sigma)

    def _drift_at(self, t: float) -> float:
        """Slow AR(1) log-bandwidth drift, indexed by *time* so that every policy run with the
        same seed sees the same environment (common random numbers)."""
        L = self.link
        dt = 0.05
        k = int(t / dt)
        g = self._drift_grid
        while len(g) <= k:
            prev = g[-1] if g else 0.0
            g.append(L.drift_rho * prev + math.sqrt(1 - L.drift_rho ** 2) * L.drift_sigma * self._drift_rng.gauss(0, 1))
        return g[k]

    def _service(self, job: Job) -> float:
        L = self.link
        if not self.det:
            f = math.exp(self._drift_at(job.t_start)) * self._noise(L.jitter_sigma)
        else:
            f = 1.0
        base = L.nominal_time(job.nbytes)
        fixed = (L.latency_us + L.sync_us) * 1e-6
        return fixed + (base - fixed) * f

    def _try_start(self, now: float):
        if self.cur is not None or not self.pending:
            return
        self.pending.sort(key=lambda j: (j.prio, j.seq))
        job = self.pending.pop(0)
        job.t_start = max(now, job.t_issue)
        dur = self._service(job)
        job.t_end = job.t_start + dur
        self.link_busy += dur
        self.cur = job

    def _complete(self, job: Job):
        job.done = True
        self.bytes_fetched += job.nbytes
        if not job.gather:
            self.state[job.uid] = 2
            self.ready[job.uid] = job.t_end
            self.last_use[job.uid] = max(self.last_use[job.uid], job.t_end)

    def advance_link(self, T: float):
        """Process all link completions with end time <= T (calling policy hooks)."""
        while self.cur is not None and self.cur.t_end <= T:
            job = self.cur
            self.cur = None
            self._complete(job)
            tc = job.t_end
            self.now = tc
            self.policy.tick(self, tc, "link")
            self._try_start(tc)

    # ----------------------------------------------------------------- cache
    def is_present(self, uid: int) -> bool:
        return self.state[uid] != 0

    def evictable(self):
        return [u for u, s in self.state.items()
                if s == 2 and u not in self.pinned and u not in self.protect]

    def _evict(self, uid: int):
        self.state[uid] = 0
        self.used -= self.units[uid].nbytes
        if self.useful_pf.get(uid) is False:
            self.prefetch_wasted += 1
        self.useful_pf.pop(uid, None)

    def make_room(self, nbytes: float, now: float, admit_before: Optional[int]) -> bool:
        while self.used + nbytes > self.cap + 1e-6:
            v = self.policy.choose_victim(self, now, admit_before)
            if v is None:
                return False
            self._evict(v)
        return True

    def issue(self, uid: int, now: float, prio: int, admit_before: Optional[int] = None) -> bool:
        """Start fetching ``uid`` (full unit).  Returns False if no room could be made."""
        if self.state[uid] != 0:
            return True
        if prio == 1 and self.demand_blocked:
            return False
        nb = self.units[uid].nbytes
        if nb > self.cap + 1e-6:
            raise RuntimeError("unit larger than cache")
        if not self.make_room(nb, now, admit_before):
            return False
        self.state[uid] = 1
        self.used += nb
        self._seq += 1
        self.ins_seq[uid] = self._seq
        job = Job(uid, nb, prio, self._seq, False, now)
        self.pending.append(job)
        if prio == 1:
            self.prefetch_issued += 1
            self.useful_pf[uid] = False
        self._try_start(now)
        return True

    def promote(self, uid: int):
        for j in self.pending:
            if j.uid == uid and not j.gather:
                j.prio = 0

    def preload_pins(self, pins):
        self.pinned = set(pins)
        total = 0.0
        for u in self.pinned:
            total += self.units[u].nbytes
            self.state[u] = 2
            self.ready[u] = -INF
        if total > self.cap + 1e-6:
            raise RuntimeError(f"pinned set ({total/1e9:.2f} GB) exceeds cache ({self.cap/1e9:.2f} GB)")
        self.used = total

    # ------------------------------------------------------------------- run
    def run(self) -> Result:
        tr = self.trace
        acc = tr.acc
        self.policy.setup(self)
        self.preload_pins(self.policy.pins)
        t = 0.0
        iter_times: List[float] = []
        iter_stall: List[float] = []
        it_start = 0.0
        it_stall = 0.0
        cur_it = acc[0].it if acc else 0
        gap = self.gpu.sampling_gap_us * 1e-6
        self.now = 0.0
        self.i = 0
        self.computing = False
        self.policy.tick(self, 0.0, "start")
        for i, a in enumerate(acc):
            if a.it != cur_it:
                # iteration boundary: sampling/scheduling gap, link keeps working
                iter_times.append(t - it_start)
                iter_stall.append(it_stall)
                t += gap
                it_start = t
                it_stall = 0.0
                cur_it = a.it
            self.i = i
            self.computing = False
            self.protect = {a.uid}
            self.advance_link(t)
            self.now = t
            t_arr = t
            uid = a.uid
            gjob = None
            # (1) the demand for the layer about to execute is issued *before* any lookahead, so
            #     prefetches can never take the room that this layer needs
            if a.fetch_bytes is not None and self.state[uid] == 0:
                self._seq += 1
                gjob = Job(uid, a.fetch_bytes, 0, self._seq, True, t)
                self.pending.append(gjob)
                self._try_start(t)
                self.demand_fetches += 1
            elif a.fetch_bytes is None:
                st = self.state[uid]
                if st == 2:
                    self.hits += 1
                    if uid in self.useful_pf:
                        self.useful_pf[uid] = True
                elif st == 0:
                    self.demand_fetches += 1
                    self.demand_blocked = True
                    while not self.issue(uid, t, 0):
                        if self.cur is None:
                            raise RuntimeError("deadlock: no room and no transfer in flight")
                        tn = self.cur.t_end
                        self.advance_link(tn)
                        t = max(t, tn)
                    self.demand_blocked = False
                else:
                    self.late_prefetches += 1
                    self.promote(uid)
                    if uid in self.useful_pf:
                        self.useful_pf[uid] = True
            # (2) lookahead
            self.policy.tick(self, t, "begin")
            # (3) wait until the layer's weights are present
            if gjob is not None:
                while not gjob.done:
                    if self.cur is None:
                        self._try_start(t)
                        if self.cur is None:
                            raise RuntimeError("deadlock: gather with idle link")
                    self.advance_link(self.cur.t_end)
                t = max(t, gjob.t_end)
            elif a.fetch_bytes is None:
                while self.state[uid] != 2:
                    if self.cur is None:
                        self._try_start(t)
                        if self.cur is None:
                            raise RuntimeError("deadlock: waiting for unit with idle link")
                    tn = self.cur.t_end
                    self.advance_link(tn)
                    t = max(t, tn)
                t = max(t, self.ready[uid])
            stall = t - t_arr
            self.stall += stall
            it_stall += stall
            # ---- compute
            self.protect = {uid}
            self.computing = True
            self.cur_start = t
            self.now = t
            self.advance_link(t)
            self.policy.tick(self, t, "compute")
            d = a.t_true * self._noise(self.compute_sigma)
            self.advance_link(t + d)
            t += d
            self.comp += d
            self.last_use[uid] = t
            self.freq[uid] += 1
            self.protect = set()
            self.policy.observe(self, a, d)
        iter_times.append(t - it_start)
        iter_stall.append(it_stall)
        return Result(total_time=t, iter_times=iter_times, stall_time=self.stall, compute_time=self.comp,
                      link_busy=self.link_busy, bytes_fetched=self.bytes_fetched,
                      demand_fetches=self.demand_fetches, late_prefetches=self.late_prefetches,
                      hits=self.hits, accesses=len(acc), prefetch_issued=self.prefetch_issued,
                      prefetch_wasted=self.prefetch_wasted, iter_stall=iter_stall)
