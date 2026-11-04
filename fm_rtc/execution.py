"""Deterministic simulation of RTC Algorithm 1 with a fixed inference delay.

Compute a pending plan from the observation at inference start, continue the old
plan for d environment steps, then activate the new plan at index d. Wall-clock
inference duration is recorded separately, never silently mapped to physics time.
"""
import time


class ChunkController:
    def __init__(self, policy, execute_steps=8, delay_steps=0, rtc=True, generator=None):
        if not 0 <= delay_steps <= execute_steps <= policy.horizon - delay_steps:
            raise ValueError("Require delay <= execute_steps <= horizon-delay")
        if execute_steps < 1:
            raise ValueError("execute_steps must be positive")
        self.policy, self.s, self.d = policy, execute_steps, delay_steps
        self.rtc, self.generator = rtc, generator
        self.current = self.pending = None
        self.origin = self.pending_origin = 0
        self.next_start = 0
        self.step = 0
        self.plan_ms = []

    def _plan(self, obs):
        start = time.perf_counter()
        plan = self.policy.plan(obs, executed=self.s, delay=self.d,
                                rtc=self.rtc, generator=self.generator)
        self.plan_ms.append(1000 * (time.perf_counter() - start))
        return plan

    def action(self, obs):
        seam = False
        if self.current is None:
            self.policy.reset()
            self.current = self._plan(obs)  # warm-up before the episode clock starts
            self.next_start = self.s
        if self.pending is not None and self.step == self.pending_origin + self.d:
            self.current, self.origin = self.pending, self.pending_origin
            self.pending = None
            seam = True
        if self.step == self.next_start:
            self.pending = self._plan(obs)
            self.pending_origin = self.step
            self.next_start += self.s
            if self.d == 0:
                self.current, self.origin = self.pending, self.pending_origin
                self.pending = None
                seam = True
        index = self.step - self.origin
        if not 0 <= index < len(self.current):
            raise RuntimeError("Action buffer underrun")
        action = self.current[index, :self.policy.act_dim].copy()
        self.step += 1
        return action, seam
