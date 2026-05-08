from __future__ import annotations

import struct
import sys
from typing import Iterable, List

from SDK.backend.forecast import build_forecast_state
from SDK.backend.model import Operation
from SDK.backend.runtime import MatchRuntime
from SDK.backend.engine import PublicRoundState
from SDK.utils.constants import OperationType

try:
    from common import MatchSession
except ModuleNotFoundError:
    from AI.common import MatchSession


def _forecast_to_model(ops) -> List[Operation]:
    return [Operation(op.type, op.arg0, op.arg1) for op in ops]


class _IO:
    def __init__(self):
        self.stdin = sys.stdin.buffer
        self.stdout = sys.stdout.buffer

    def recv_line(self):
        raw = self.stdin.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n")

    def send_packet(self, payload: str) -> None:
        if not payload.endswith("\n"):
            payload += "\n"
        data = payload.encode("utf-8")
        self.stdout.write(struct.pack(">I", len(data)))
        self.stdout.write(data)
        self.stdout.flush()

    def recv_init(self):
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing init line")
        player, seed = map(int, line.split())
        return player, seed

    def recv_operations(self):
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing operation count")
        count = int(line.strip())
        operations = []
        for _ in range(count):
            payload = self.recv_line()
            if payload is None:
                raise RuntimeError("unexpected EOF")
            parts = [int(item) for item in payload.split()]
            op_type = OperationType(parts[0])
            if len(parts) == 1:
                operations.append(Operation(op_type))
            elif len(parts) == 2:
                operations.append(Operation(op_type, parts[1]))
            else:
                operations.append(Operation(op_type, parts[1], parts[2]))
        return operations

    def recv_round_state(self):
        line = self.recv_line()
        if line is None:
            return None
        round_index = int(line.strip())
        tower_count = int((self.recv_line() or "0").strip())
        towers = []
        for _ in range(tower_count):
            towers.append(tuple(map(int, (self.recv_line() or "").split())))
        ant_count = int((self.recv_line() or "0").strip())
        ants = []
        for _ in range(ant_count):
            ants.append(tuple(map(int, (self.recv_line() or "").split())))
        coins = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        camp_fields = tuple(map(int, (self.recv_line() or "0 0").split()))
        camps_hp = camp_fields[:2]
        speed_lv = camp_fields[2:4] if len(camp_fields) >= 4 else None
        anthp_lv = camp_fields[4:6] if len(camp_fields) >= 6 else None
        cooldown_row_count = int((self.recv_line() or "0").strip())
        weapon_cooldowns = []
        for _ in range(cooldown_row_count):
            weapon_cooldowns.append(tuple(map(int, (self.recv_line() or "").split())))
        active_effect_count = int((self.recv_line() or "0").strip())
        active_effects = []
        for _ in range(active_effect_count):
            active_effects.append(tuple(map(int, (self.recv_line() or "").split())))
        return PublicRoundState(
            round_index=round_index,
            towers=towers,
            ants=ants,
            coins=coins,
            camps_hp=camps_hp,
            speed_lv=tuple(speed_lv) if speed_lv is not None else None,
            anthp_lv=tuple(anthp_lv) if anthp_lv is not None else None,
            weapon_cooldowns=tuple(weapon_cooldowns),
            active_effects=active_effects,
        )

    def send_operations(self, operations: Iterable[Operation]) -> None:
        items = list(operations)
        lines = [str(len(items))]
        lines.extend(
            " ".join(str(t) for t in op.to_protocol_tokens())
            for op in items
        )
        self.send_packet("\n".join(lines) + "\n")


class ApexSession(MatchSession):
    def __init__(self, ai) -> None:
        self.io = _IO()
        player, seed = self.io.recv_init()
        self._player = player
        self.runtime = MatchRuntime.create(
            player=player, seed=seed, prefer_native=False,
        )
        self.ai = ai

    @property
    def player(self) -> int:
        return self._player

    def perform_self_turn(self) -> None:
        game_info = build_forecast_state(self.runtime.state)
        forecast_ops = self.ai(self._player, game_info)
        model_ops = _forecast_to_model(forecast_ops)
        accepted: list[Operation] = []
        for op in model_ops:
            if self.runtime.state.can_apply_operation(self._player, op, accepted):
                accepted.append(op)
        self.runtime.apply_self_operations(accepted)
        self.io.send_operations(accepted)

    def receive_opponent_turn(self) -> bool:
        try:
            ops = self.io.recv_operations()
        except Exception:
            return False
        self.runtime.apply_opponent_operations(ops)
        return True

    def sync_round(self) -> bool:
        round_state = self.io.recv_round_state()
        if round_state is None:
            return False
        self.runtime.finish_round(round_state)
        return True
