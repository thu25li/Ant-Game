from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence, Tuple

from SDK.backend.forecast import (
    MAX_ROUND,
    MAP_PROPERTY,
    Ant,
    AntState,
    BuildingType,
    ForecastOperation as Operation,
    ForecastSimulator as Simulator,
    ForecastState as GameInfo,
    OperationType,
    SuperWeaponType,
    Tower,
    TowerType,
    hex_distance as distance,
    is_valid_pos,
)
from SDK.utils.constants import ANT_AGE_LIMIT
from SDK.utils.constants import (
    BASE_HP,
    BASE_UPGRADE_COST,
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
    SUPER_WEAPON_STATS,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    tower_build_cost_for_count,
    AntKind,
)

SEARCH_BUDGET = 0.28
MAX_NODE_COUNT = 18000
SEARCH_STAGING_ENEMY_BASE_HP = BASE_HP
EVALUATION_HORIZON = 48
HOME_SLOT = 0
STORM_SLOT = 34
EMP_COST = SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost
EMP_COOLDOWN = SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown
DEFLECTOR_COST = SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost
EVASION_COST = SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost
STORM_COST = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
EMP_BUFFER_CAP = max(EMP_COST - 1, 0)
LEVEL2_BASE_UPGRADE_COST, LEVEL3_BASE_UPGRADE_COST = BASE_UPGRADE_COST
LEVEL2_TOWER_TOTAL_COST = LEVEL2_TOWER_UPGRADE_COST
LEVEL3_TOWER_TOTAL_COST = LEVEL2_TOWER_UPGRADE_COST + LEVEL3_TOWER_UPGRADE_COST
GLOBAL_TURN_START = 0.0
GLOBAL_TIME_LIMIT = 4.0


def _check_time() -> bool:
    return time.time() - GLOBAL_TURN_START < GLOBAL_TIME_LIMIT


SITE_LAYOUT = (
    (
        (2, 9), (4, 9), (5, 9), (5, 7), (6, 9), (5, 11), (5, 6), (6, 7),
        (6, 11), (5, 12), (4, 3), (5, 3), (7, 8), (7, 10), (4, 15), (5, 15),
        (4, 2), (6, 4), (7, 5), (8, 7), (8, 11), (7, 13), (6, 14), (4, 16),
        (6, 1), (6, 2), (6, 16), (6, 17), (7, 1), (8, 4), (8, 14), (7, 17),
        (8, 2), (8, 16), (3, 9),
    ),
    (
        (16, 9), (14, 9), (13, 9), (13, 7), (12, 9), (13, 11), (12, 6), (12, 7),
        (12, 11), (12, 12), (14, 3), (13, 3), (10, 8), (10, 10), (14, 15), (13, 15),
        (13, 2), (11, 4), (11, 5), (10, 7), (10, 11), (11, 13), (11, 14), (13, 16),
        (12, 1), (11, 2), (11, 16), (12, 17), (11, 1), (9, 4), (9, 14), (11, 17),
        (9, 2), (9, 16), (15, 9),
    ),
)

ACTIONABLE_SITES = (
    1, 2, 4, 10, 16, 11, 14, 23, 15, 17, 18, 22, 21,
    3, 6, 7, 5, 8, 9, 19, 12, 13, 20, 24, 25, 28, 27, 26, 31,
)

SITE_FAMILIES = (
    (1, 2, 4), (3, 6, 7), (5, 8, 9), (10, 16, 11),
    (14, 23, 15), (19, 12, 13, 20), (17, 18), (22, 21),
    (24, 25, 28), (27, 26, 31), (32, 29), (30, 33),
)

SITE_TO_FAMILY: dict[int, tuple[int, ...]] = {}
for _fam in SITE_FAMILIES:
    for _s in _fam:
        SITE_TO_FAMILY[_s] = _fam

T2_UPGRADE_ORDER = (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR)
T3_FROM_HEAVY = (TowerType.BEWITCH, TowerType.ICE, TowerType.HEAVY_PLUS)
T3_FROM_QUICK = (TowerType.SNIPER, TowerType.QUICK_PLUS, TowerType.DOUBLE)
T3_FROM_MORTAR = (TowerType.MISSILE, TowerType.MORTAR_PLUS, TowerType.PULSE)


def _total_build_investment(tower_count: int) -> int:
    return sum(tower_build_cost_for_count(i) for i in range(max(tower_count, 0)))


def _tower_dps(tower_type: TowerType) -> float:
    s = TOWER_STATS[tower_type]
    if s.damage <= 0 or s.speed <= 0:
        return 0.0
    return s.damage / s.speed


def _load_runtime_module():
    module_name = "_agent_v11_runtime"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    module_path = Path(__file__).with_name("runtime.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("unable to load runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ForecastNode:
    __slots__ = (
        "brain", "sim", "node_id", "parent", "children", "chosen", "score",
        "best_descendant", "round_tag", "sunk_cost", "best_depth",
        "expanded_layers", "collapse_round", "danger", "solvent",
        "distance_trace",
    )

    def __init__(self, brain: AI, sim: Simulator) -> None:
        self.brain = brain
        self.sim = sim.clone()
        self.node_id = -1
        self.parent = -1
        self.children: List[int] = []
        self.chosen: List[Operation] = []
        self.score = 0.0
        self.best_descendant = 0.0
        self.round_tag = self.sim.info.round
        self.sunk_cost = 0.0
        self.best_depth = 0
        self.expanded_layers = 0
        self.collapse_round = 0
        self.danger = False
        self.solvent = True
        self.distance_trace = [0] * EVALUATION_HORIZON

    @property
    def action_count(self) -> int:
        return len(self.chosen)

    def _record_hostile_distance(self, info: GameInfo) -> None:
        brain = self.brain
        idx = info.round - brain.current_round
        if 0 <= idx < EVALUATION_HORIZON:
            self.distance_trace[idx] = brain._nearest_hostile_step(info)

    def _advance_until_hp_drop(self, trial: Simulator, hp_drop: int) -> int:
        info = trial.info
        brain = self.brain
        horizon = brain.current_round + EVALUATION_HORIZON
        for turn in range(info.round, horizon):
            if not trial.fast_next_round(brain.side):
                break
            self.distance_trace[turn - brain.current_round] = brain._nearest_hostile_step(info)
            if info.bases[brain.side].hp <= brain.wall_hp_snapshot - hp_drop:
                return info.round
        return horizon

    def _forecast_ruin_round(self, trial: Simulator) -> int:
        info = trial.info
        brain = self.brain
        horizon = brain.current_round + EVALUATION_HORIZON
        ruin_round = horizon
        if info.bases[brain.side].hp <= brain.wall_hp_snapshot - 1:
            ruin_round = self.collapse_round
        elif info.bases[brain.side].hp <= brain.wall_hp_snapshot - 2:
            self.collapse_round = self._advance_until_hp_drop(trial, 1)
            ruin_round = self.collapse_round
        else:
            ruin_round = self._advance_until_hp_drop(trial, 2)
        return ruin_round

    def _safe_gap(self, info: GameInfo) -> int:
        brain = self.brain
        if brain.current_round <= 60:
            self.solvent = True
            return 0
        gap = brain._cash_safety_gap(info)
        self.solvent = gap == 0
        return gap

    def _score_survival(self, info: GameInfo, ruin_round: int) -> float:
        brain = self.brain
        hp_delta = info.bases[brain.side].hp - brain.wall_hp_snapshot
        rounds_safe = self.collapse_round - brain.current_round
        buffer = ruin_round - self.collapse_round
        w = 1.5 if rounds_safe < 16 else (1.25 if rounds_safe < 30 else 1.0)
        return hp_delta * 2.0 + rounds_safe * 0.8 * w + buffer * 0.1 - self.sunk_cost * 2.5 + 20

    def _score_frontline(self, info: GameInfo) -> float:
        brain = self.brain
        if brain.front_state != 0:
            return 0.0
        aw = {0: 3.0, 1: 5.0, 2: 7.0}[info.bases[1 - brain.side].ant_level]
        return (
            -(info.old_count[1 - brain.side] - brain.enemy_old_baseline) * aw * 2.0
            + (info.die_count[1 - brain.side] - brain.enemy_die_baseline) * aw * 1.5
        )

    def _score_danger(self, ruin_round: int) -> float:
        brain = self.brain
        self.danger = False
        gap = self.collapse_round - brain.current_round
        if gap > 16:
            return 0.0
        self.danger = True
        score = -500.0
        if ruin_round - self.collapse_round <= 8:
            score -= 300.0
        if gap <= 4:
            score -= 200.0
        return score

    def _score_cash_safety(self, safe_gap: int) -> float:
        brain = self.brain
        if self.solvent or self.danger or brain.front_state < 0:
            return 0.0
        return (-40 + safe_gap / 5) * min((brain.current_round - 60) / 30, 1.0)

    def _my_towers(self, info: GameInfo) -> List[Tower]:
        return [t for t in info.towers if t.player == self.brain.side]

    def _score_tower_quality(self, towers: Sequence[Tower]) -> float:
        if not towers:
            return 0.0
        tc = len(towers)
        round_ratio = self.brain.current_round / 512.0
        score = tc * 0.8
        midgame_penalty_start = 7 if round_ratio < 0.3 else 9
        if tc > midgame_penalty_start:
            score -= (tc - midgame_penalty_start) * 2.0
        if tc > 12:
            score -= (tc - 12) * 4.0
        score -= _total_build_investment(tc) * 0.08
        for tower in towers:
            tier = int(tower.type) // 10
            if 0 < int(tower.type) and tier == 0:
                score -= LEVEL2_TOWER_TOTAL_COST * 0.15
                score += _tower_dps(tower.type) * 2.0
            elif tier > 0:
                score -= LEVEL3_TOWER_TOTAL_COST * 0.15
                score += _tower_dps(tower.type) * 3.0
                tt = tower.type
                if tt == TowerType.ICE: score += 12.0
                elif tt == TowerType.BEWITCH: score += 14.0
                elif tt == TowerType.MISSILE: score += 8.0
                elif tt == TowerType.PULSE: score += 7.0
                elif tt == TowerType.MORTAR_PLUS: score += 6.0
                elif tt == TowerType.DOUBLE: score += 5.0
                elif tt == TowerType.SNIPER: score += 5.0
                elif tt == TowerType.HEAVY_PLUS: score += 4.0
                elif tt == TowerType.QUICK_PLUS: score += 3.0
            else:
                score += _tower_dps(tower.type) * 1.0
        return score

    def _score_tower_spacing(self, towers: Sequence[Tower]) -> float:
        tc = len(towers)
        if tc <= 1:
            return 0.0
        penalty = 0.0
        distanced = False
        for idx, t in enumerate(towers[:-1]):
            for o in towers[idx + 1:]:
                g = distance(t.x, t.y, o.x, o.y)
                if g <= 3: penalty += 5
                elif g <= 6: penalty += 2
                else: distanced = True
        if tc >= 3 and not distanced:
            penalty += 20
        return -penalty / math.sqrt(tc)

    def _score_tower_advancement(self, towers: Sequence[Tower], info: GameInfo) -> float:
        base = info.bases[self.brain.side]
        return sum(distance(t.x, t.y, base.x, base.y) * 0.4 for t in towers)

    @staticmethod
    def _world_pos(x: int, y: int) -> Tuple[float, float]:
        return x + 0.5 * (y & 1), y * math.sqrt(3) / 2

    @staticmethod
    def _angle_delta(angle: float, target: float) -> float:
        return (angle - target + math.pi) % (2 * math.pi) - math.pi

    def _score_arc_coverage(self, towers: Sequence[Tower], info: GameInfo) -> float:
        base = info.bases[self.brain.side]
        eb = info.bases[1 - self.brain.side]
        bx, by = self._world_pos(base.x, base.y)
        ex, ey = self._world_pos(eb.x, eb.y)
        fa = math.atan2(ey - by, ex - bx)
        tol = math.radians(20.0)
        targets = (-30.0, 0.0, 30.0)
        covered = {t: False for t in targets}
        for tower in towers:
            tx, ty = self._world_pos(tower.x, tower.y)
            a = math.atan2(ty - by, tx - bx)
            if abs(self._angle_delta(a, fa)) > math.pi / 2:
                continue
            for t in targets:
                if abs(self._angle_delta(a, fa + math.radians(t))) <= tol:
                    covered[t] = True
        return -sum(8.0 for c in covered.values() if not c)

    def _score_hostile_trace(self, info: GameInfo) -> float:
        brain = self.brain
        if brain.front_state < 0:
            return 0.0
        score = 0.0
        close = False
        for idx in range(min(EVALUATION_HORIZON, info.round - brain.current_round - 4)):
            d = self.distance_trace[idx]
            if d <= 3: close = True
            if d == 5: score -= 0.2
            elif d == 4: score -= 2.5
            elif d in (1, 2, 3): score -= 2.0
        if close:
            score -= 20
        return score

    def _score_enemy_pressure(self, info: GameInfo) -> float:
        brain = self.brain
        if brain.front_state < 0 or brain.current_round < 20:
            return 0.0
        base = info.bases[brain.side]
        ec = 0
        pressure = 0.0
        for ant in info.ants:
            if ant.player != 1 - brain.side:
                continue
            pressure += ANT_AGE_LIMIT - ant.age - distance(ant.x, ant.y, base.x, base.y) * 1.5
            ec += 1
        if ec == 0:
            return 0.0
        return pressure / ec * 0.5

    def _score_offensive_pressure(self, info: GameInfo) -> float:
        brain = self.brain
        if brain.current_round < 20:
            return 0.0
        enemy_base = info.bases[1 - brain.side]
        my_ants = [a for a in info.ants if a.player == brain.side]
        if not my_ants:
            return 0.0
        front_dist = min(distance(a.x, a.y, enemy_base.x, enemy_base.y) for a in my_ants)
        score = max(0.0, 20.0 - front_dist) * 0.8
        near_ants = sum(1 for a in my_ants if distance(a.x, a.y, enemy_base.x, enemy_base.y) <= 6)
        score += near_ants * 1.5
        return score

    def _score_enemy_towers_destroyed(self, info: GameInfo) -> float:
        brain = self.brain
        enemy = 1 - brain.side
        enemy_tower_count = sum(1 for t in info.towers if t.player == enemy)
        return (brain.initial_enemy_tower_count - enemy_tower_count) * 8.0

    def _bundle_heuristic_score(self, bundle: list[Operation], info: GameInfo) -> float:
        score = 0.0
        enemy_base = info.bases[1 - self.brain.side]
        for op in bundle:
            if op.type == OperationType.BUILD_TOWER:
                dist = distance(op.arg0, op.arg1, enemy_base.x, enemy_base.y)
                score += max(0.0, 10.0 - dist) * 1.2
                score += 3.0
            elif op.type == OperationType.UPGRADE_TOWER:
                tower = self.brain._tower_by_id(op.arg0, info)
                if tower is not None:
                    if tower.type == TowerType.BASIC:
                        score += 10.0
                    else:
                        score += 6.0
            elif op.type == OperationType.DOWNGRADE_TOWER:
                score -= 2.5
            elif op.type == OperationType.UPGRADE_GENERATED_ANT:
                score += 8.0
            elif op.type == OperationType.UPGRADE_GENERATION_SPEED:
                score += 6.0
            elif op.type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER):
                score += 8.0
            elif op.type in (OperationType.USE_DEFLECTOR, OperationType.USE_EMERGENCY_EVASION):
                score += 5.5
        return score

    def evaluate(self) -> float:
        trial = self.sim.clone()
        info = trial.info
        self._record_hostile_distance(info)
        safe_gap = self._safe_gap(info)
        ruin_round = self._forecast_ruin_round(trial)
        my_towers = self._my_towers(info)
        score = 0.0
        score += self._score_survival(info, ruin_round)
        score += self._score_frontline(info)
        score += self._score_danger(ruin_round)
        score += self._score_cash_safety(safe_gap)
        score += self._score_tower_quality(my_towers)
        score += self._score_tower_spacing(my_towers)
        score += self._score_tower_advancement(my_towers, info)
        score += self._score_arc_coverage(my_towers, info)
        score += self._score_hostile_trace(info)
        score += self._score_enemy_pressure(info)
        score += self._score_offensive_pressure(info)
        score += self._score_enemy_towers_destroyed(info)
        self.score = score
        self.best_descendant = score
        return score

    def expand(self, is_root: bool = False) -> None:
        brain = self.brain
        info = self.sim.info
        if info.round >= MAX_ROUND or info.bases[brain.side].hp <= 0 or info.bases[1 - brain.side].hp <= 0:
            return
        if not is_root:
            if info.round - brain.current_round < EVALUATION_HORIZON:
                self.distance_trace[info.round - brain.current_round] = brain._nearest_hostile_step(info)
            if not self.sim.fast_next_round(brain.side):
                return

        emp_blocked = [False] * 34
        for weapon in self.sim.info.super_weapons:
            if weapon.player == 1 - brain.side and weapon.type == SuperWeaponType.EMP_BLASTER:
                for site in range(34):
                    sx, sy = SITE_LAYOUT[brain.side][site]
                    if distance(weapon.x, weapon.y, sx, sy) <= 3:
                        emp_blocked[site] = True
                break

        bundles: List[List[Operation]] = []
        for tactic in range(8):
            if not _check_time():
                break
            if self.action_count > 0 and tactic in (3, 5):
                continue
            if (
                self.action_count == 1
                and self.chosen[0].type == OperationType.BUILD_TOWER
                and self.expanded_layers < 2
                and tactic in (3, 4, 6)
            ):
                continue
            if (
                self.action_count == 1
                and self.chosen[0].type == OperationType.UPGRADE_TOWER
                and self.expanded_layers < 2
                and tactic == 2
            ):
                continue
            if (
                self.action_count == 2
                and self.chosen[1].type == OperationType.BUILD_TOWER
                and self.expanded_layers < 2
                and tactic in (3, 4, 6)
            ):
                continue
            if self.sim.info.tower_num_of_player(brain.side) >= 7 and tactic in (0, 2):
                continue
            bundles.extend(brain._candidate_bundles(tactic, self.sim.info, emp_blocked))

        if bundles:
            bundles.sort(key=lambda bundle: self._bundle_heuristic_score(bundle, info), reverse=True)
            bundles = bundles[:120]

        if is_root:
            idle = ForecastNode(brain, self.sim)
            idle.node_id = len(brain.nodes)
            idle.parent = self.node_id
            idle.evaluate()
            brain.nodes.append(idle)
            self.children.append(idle.node_id)

        for bundle in bundles:
            if len(brain.nodes) >= MAX_NODE_COUNT - 10:
                break
            child = ForecastNode(brain, self.sim)
            child.node_id = len(brain.nodes)
            child.parent = self.node_id
            child.sunk_cost = self.sunk_cost
            child.chosen = list(bundle)
            child.collapse_round = self.collapse_round
            child.score = -1e9
            child.best_descendant = -1e9
            if self.sim.info.round > brain.current_round:
                tl = min(EVALUATION_HORIZON, self.sim.info.round - brain.current_round)
                child.distance_trace[:tl] = self.distance_trace[:tl]
            child.sim.operations[0].clear()
            child.sim.operations[1].clear()
            mutable = child.sim.info
            for op in bundle:
                if op.type == OperationType.DOWNGRADE_TOWER:
                    tower = brain._tower_by_id(op.arg0, mutable)
                    if tower is not None:
                        if tower.type == TowerType.BASIC:
                            child.sunk_cost += mutable.build_tower_cost(mutable.tower_num_of_player(brain.side)) * 0.2
                        else:
                            child.sunk_cost += mutable.upgrade_tower_cost(int(tower.type)) * 0.2
                child.sim.add_operation_of_player(brain.side, op)
            child.sim.apply_operations_of_player(brain.side)
            value = child.evaluate()
            if value > self.best_descendant:
                self.best_descendant = value
                self.best_depth = self.expanded_layers + 1
            brain.nodes.append(child)
            self.children.append(child.node_id)

        if is_root and not self.sim.fast_next_round(brain.side):
            return
        self.expanded_layers += 1


class AI:
    def __init__(self) -> None:
        self.side = 0
        self.current_round = 0
        self.front_state = 0
        self.wall_hp_snapshot = 0
        self.enemy_old_baseline = 0
        self.enemy_die_baseline = 0
        self.initial_enemy_tower_count = 0
        self.assault_memory = False
        self.last_superweapon_type: Optional[SuperWeaponType] = None
        self.last_superweapon_round = -1
        self.reserve_depth = 0
        self.nodes: List[ForecastNode] = []

    def create_session(self):
        return _load_runtime_module().ApexSession(self)

    def _mark_super(self, weapon_type: SuperWeaponType) -> None:
        self.last_superweapon_round = self.current_round
        self.last_superweapon_type = weapon_type

    def _tower_at(self, x: int, y: int, info: GameInfo) -> Optional[Tower]:
        for t in info.towers:
            if t.x == x and t.y == y:
                return t
        return None

    def _tower_by_id(self, tower_id: int, info: GameInfo) -> Optional[Tower]:
        return info.tower_of_id(tower_id)

    def _nearest_push_distance(self, info: GameInfo) -> int:
        best = 100
        tx, ty = SITE_LAYOUT[1 - self.side][HOME_SLOT]
        for ant in info.ants:
            if ant.player == self.side:
                best = min(best, distance(ant.x, ant.y, tx, ty))
        return best

    def _opponent_emp_buffer(self, info: GameInfo) -> int:
        cd = info.super_weapon_cd[1 - self.side][int(SuperWeaponType.EMP_BLASTER)]
        if cd >= EMP_COOLDOWN - 10:
            return 0
        if cd > 0:
            return max(int(min(info.coins[1 - self.side], EMP_BUFFER_CAP) - cd * 1.66), 0)
        return min(info.coins[1 - self.side], EMP_BUFFER_CAP)

    def _cash_safety_gap(self, info: GameInfo) -> int:
        return max(0, info.coins[self.side] - self._opponent_emp_buffer(info))

    def _nearest_hostile_step(self, info: GameInfo) -> int:
        best = 32
        tx, ty = SITE_LAYOUT[self.side][HOME_SLOT]
        for ant in info.ants:
            if ant.player == 1 - self.side:
                best = min(best, distance(ant.x, ant.y, tx, ty))
        return best

    def _site_operation(
        self, site: int, mode: int, info: GameInfo, coins: int, towers: int,
        upgrade_branch: int = 0, exempt_site: int = -1,
    ) -> Tuple[Optional[Operation], int, int]:
        x, y = SITE_LAYOUT[self.side][site]
        if mode == 1:
            cost = info.build_tower_cost(towers)
            if coins < cost:
                return None, coins, towers
            for peer in SITE_TO_FAMILY[site]:
                if peer == exempt_site:
                    continue
                px, py = SITE_LAYOUT[self.side][peer]
                if info.building_tag[px][py] != BuildingType.EMPTY:
                    return None, coins, towers
            return Operation(OperationType.BUILD_TOWER, x, y), coins - cost, towers + 1
        if mode == 2:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or int(tower.type) // 10 > 0:
                return None, coins, towers
            if tower.type == TowerType.BASIC:
                target = T2_UPGRADE_ORDER[upgrade_branch]
            elif tower.type == TowerType.HEAVY:
                target = T3_FROM_HEAVY[upgrade_branch]
            elif tower.type == TowerType.QUICK:
                target = T3_FROM_QUICK[upgrade_branch]
            elif tower.type == TowerType.MORTAR:
                target = T3_FROM_MORTAR[upgrade_branch]
            else:
                return None, coins, towers
            cost = info.upgrade_tower_cost(int(target))
            if coins < cost:
                return None, coins, towers
            return Operation(OperationType.UPGRADE_TOWER, tower.id, int(target)), coins - cost, towers
        if mode == 3:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or tower.type != TowerType.BASIC:
                return None, coins, towers
            refund = info.destroy_tower_income(towers)
            return Operation(OperationType.DOWNGRADE_TOWER, tower.id), coins + refund, towers - 1
        if mode == 4:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or tower.type == TowerType.BASIC:
                return None, coins, towers
            refund = info.downgrade_tower_income(int(tower.type))
            return Operation(OperationType.DOWNGRADE_TOWER, tower.id), coins + refund, towers
        return None, coins, towers

    def _candidate_bundles(self, tactic: int, info: GameInfo, emp_blocked: Sequence[bool]) -> List[List[Operation]]:
        bundles: List[List[Operation]] = []
        if tactic == 0:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 1, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 1:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                for branch in range(3):
                    op, _, _ = self._site_operation(site, 2, info, info.coins[self.side], info.tower_num_of_player(self.side), branch)
                    if op is not None:
                        bundles.append([op])
        elif tactic == 2:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(site, 4, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    tail, _, _ = self._site_operation(site2, 1, info, coins, towers)
                    if tail is not None:
                        bundles.append([head, tail])
        elif tactic == 3:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 3, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 4:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(site, 3, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    for branch in range(3):
                        tail, _, _ = self._site_operation(site2, 2, info, coins, towers, branch)
                        if tail is not None:
                            bundles.append([head, tail])
        elif tactic == 5:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 4, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 6:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(site, 3, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    tail, _, _ = self._site_operation(site2, 1, info, coins, towers, exempt_site=site)
                    if tail is not None:
                        bundles.append([head, tail])
        elif tactic == 7:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(site, 4, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    for branch in range(3):
                        tail, _, _ = self._site_operation(site2, 2, info, coins, towers, branch)
                        if tail is not None:
                            bundles.append([head, tail])
        return bundles

    def _expand_one(self) -> bool:
        root = self.nodes[0]
        if not root.children:
            return False
        target_id = -1
        best = -1e9
        for child_id in root.children:
            child = self.nodes[child_id]
            value = -child.expanded_layers * 1.5
            if child_id == 0:
                value += self.reserve_depth
            if not child.children:
                value += 100
            if child.danger:
                value += 30
            if not child.solvent:
                value -= 20
            value += child.best_descendant * 0.001
            if value > best:
                best = value
                target_id = child_id
        if target_id < 0:
            return False
        self.nodes[target_id].expand()
        return True

    def _support_expand(self, bias: int) -> None:
        root = self.nodes[0]
        if not root.children:
            return
        for child_id in root.children:
            if not _check_time():
                break
            child = self.nodes[child_id]
            if child.collapse_round - self.current_round > 24:
                continue
            now_round = child.sim.info.round
            target_round = min(MAX_ROUND - 1, child.collapse_round - bias)
            if now_round >= target_round:
                continue
            for _ in range(now_round, target_round - 1):
                if not child.sim.fast_next_round(self.side):
                    break
            child.expand()

    def _max_future_liquidation_coins(self, info: GameInfo, operations: Sequence[Operation]) -> int:
        trial = info.clone()
        for op in operations:
            trial.apply_operation(self.side, op)
        while True:
            tids = [t.id for t in trial.towers if t.player == self.side and not trial.tower_under_emp(t)]
            if not tids:
                break
            progressed = False
            for tid in tids:
                tower = trial.tower_of_id(tid)
                if tower is None:
                    continue
                trial.apply_operation(self.side, Operation(OperationType.DOWNGRADE_TOWER, tid))
                progressed = True
            if not progressed:
                break
        return trial.coins[self.side]

    def _liquidate_all(self, coins: int, towers: int, coin_need: int, info: GameInfo) -> Optional[Tuple[List[Operation], int, int]]:
        ops: List[Operation] = []
        for tower in info.towers:
            if tower.player != self.side or info.tower_under_emp(tower):
                continue
            if tower.type == TowerType.BASIC:
                coins += info.destroy_tower_income(towers)
                towers -= 1
            else:
                coins += info.downgrade_tower_income(int(tower.type))
            ops.append(Operation(OperationType.DOWNGRADE_TOWER, tower.id))
            if coins >= coin_need:
                return ops, coins, towers
        if self._max_future_liquidation_coins(info, ops) >= coin_need:
            return ops, coins, towers
        return None

    def _liquidate_cautious(self, coins: int, towers: int, coin_need: int, info: GameInfo) -> Optional[Tuple[List[Operation], int, int]]:
        tower_ids = [t.id for t in info.towers if t.player == self.side and not info.tower_under_emp(t)]
        if not tower_ids:
            return None
        if len(tower_ids) > 5:
            tower_ids = tower_ids[:5]
        baseline = Simulator(info)
        fallback_round = 48
        for step in range(1, 49):
            if not baseline.fast_next_round(self.side):
                break
            if baseline.info.bases[self.side].hp < info.bases[self.side].hp:
                fallback_round = step
                break
        max_round = -1
        best_ops: List[Operation] = []
        for order in itertools.permutations(tower_ids):
            if not _check_time():
                break
            plan: List[Operation] = []
            trial = Simulator(info)
            snapshot = trial.info
            wallet = coins
            tc = towers
            valid = False
            for tid in order:
                tower = self._tower_by_id(tid, snapshot)
                if tower is None:
                    continue
                if tower.type == TowerType.BASIC:
                    wallet += snapshot.destroy_tower_income(tc)
                    tc -= 1
                else:
                    wallet += snapshot.downgrade_tower_income(int(tower.type))
                plan.append(Operation(OperationType.DOWNGRADE_TOWER, tid))
                if wallet >= coin_need:
                    valid = True
                    break
            if not valid:
                continue
            for op in plan:
                trial.add_operation_of_player(self.side, op)
            trial.apply_operations_of_player(self.side)
            window = 48
            base_hp = snapshot.bases[self.side].hp
            for step in range(1, 49):
                if not trial.fast_next_round(self.side):
                    break
                if trial.info.bases[self.side].hp < base_hp:
                    window = step
                    break
            if window > max_round:
                max_round = window
                best_ops = plan
        if max_round < min(24, fallback_round):
            return None
        return best_ops, coins, towers

    def _emp_target_positions(self, info: GameInfo) -> List[Tuple[int, int, float]]:
        enemy = 1 - self.side
        results: List[Tuple[int, int, float]] = []
        enemy_towers = [(t.x, t.y, t.type) for t in info.towers if t.player == enemy]
        if not enemy_towers:
            return results
        for x in range(19):
            for y in range(19):
                if MAP_PROPERTY[x][y] < 0:
                    continue
                value = 0.0
                for tx, ty, tt in enemy_towers:
                    if distance(tx, ty, x, y) <= 3:
                        if tt == TowerType.BASIC:
                            value += 50
                        elif int(tt) // 10 == 0:
                            value += 70
                        else:
                            value += 90
                if value >= 100:
                    results.append((x, y, value))
        results.sort(key=lambda r: -r[2])
        return results[:20]

    def _storm_target_positions(self, info: GameInfo) -> List[Tuple[int, int]]:
        enemy = 1 - self.side
        positions: List[Tuple[int, int]] = []
        enemy_ants = [(a.x, a.y) for a in info.ants if a.player == enemy and a.is_alive()]
        enemy_towers = [(t.x, t.y) for t in info.towers if t.player == enemy]
        seen = set()
        for ax, ay in enemy_ants:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    cx, cy = ax + dx, ay + dy
                    if 0 <= cx < 19 and 0 <= cy < 19 and is_valid_pos(cx, cy) and (cx, cy) not in seen:
                        seen.add((cx, cy))
                        positions.append((cx, cy))
        for tx, ty in enemy_towers:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    cx, cy = tx + dx, ty + dy
                    if 0 <= cx < 19 and 0 <= cy < 19 and is_valid_pos(cx, cy) and (cx, cy) not in seen:
                        seen.add((cx, cy))
                        positions.append((cx, cy))
        bx, by = SITE_LAYOUT[enemy][HOME_SLOT]
        if (bx, by) not in seen:
            positions.append((bx, by))
        sx, sy = SITE_LAYOUT[enemy][STORM_SLOT]
        if (sx, sy) not in seen:
            positions.append((sx, sy))
        return positions

    def _try_use_storm(self, info: GameInfo, all_in: bool) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)] > 0:
            return []
        cost = info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        prefix: List[Operation] = []
        can_cast = wallet >= cost
        if not can_cast:
            liq = self._liquidate_all(wallet, tower_count, cost, info) if all_in else self._liquidate_cautious(wallet, tower_count, cost, info)
            if liq is not None:
                prefix, wallet, tower_count = liq
                can_cast = wallet >= cost
        if not can_cast:
            return []
        targets = self._storm_target_positions(info)
        if not targets:
            return []
        best_value = -1
        best_point: Optional[Tuple[int, int]] = None
        for x, y in targets:
            if not _check_time():
                break
            trial = Simulator(info)
            for op in prefix:
                trial.add_operation_of_player(self.side, op)
            trial.add_operation_of_player(self.side, Operation(OperationType.USE_LIGHTNING_STORM, x, y))
            trial.apply_operations_of_player(self.side)
            fail_round = 24
            for tick in range(24):
                if not trial.fast_next_round(self.side):
                    break
                if trial.info.bases[self.side].hp < info.bases[self.side].hp:
                    fail_round = tick
                    break
            if fail_round < 16:
                continue
            value = trial.info.die_count[1 - self.side] + fail_round
            enemy_tower_damage = 0
            for t in trial.info.towers:
                if t.player == 1 - self.side:
                    if t.hp < TOWER_STATS[t.type].max_hp:
                        enemy_tower_damage += TOWER_STATS[t.type].max_hp - t.hp
            value += enemy_tower_damage * 2
            if value > best_value:
                best_value = value
                best_point = (x, y)
        if best_point is None:
            return []
        return [*prefix, Operation(OperationType.USE_LIGHTNING_STORM, best_point[0], best_point[1])]

    def _try_end_storm(self, info: GameInfo) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)] > 0:
            return []
        cost = info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        prefix: List[Operation] = []
        can_cast = wallet >= cost
        if not can_cast:
            liq = self._liquidate_all(wallet, tower_count, cost, info)
            if liq is not None:
                prefix, wallet, tower_count = liq
                can_cast = wallet >= cost
        if not can_cast:
            return []
        x, y = SITE_LAYOUT[1 - self.side][STORM_SLOT]
        return [*prefix, Operation(OperationType.USE_LIGHTNING_STORM, x, y)]

    def _try_use_superweapon(self, info: GameInfo) -> List[Operation]:
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        can_emp = info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER))
        can_deflect = info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR))
        can_eva = info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION))
        enemy_storm = info.super_weapon_cd[1 - self.side][int(SuperWeaponType.LIGHTNING_STORM)] == 0 and info.coins[1 - self.side] >= info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))

        prefix: List[Operation] = []
        if not can_emp and info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0:
            sale = self._liquidate_cautious(wallet, tower_count, EMP_COST, info)
            if sale is not None:
                prefix, wallet, tower_count = sale
        if not prefix and ((info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0 and not can_deflect) or (info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0 and not can_eva)):
            sale = self._liquidate_cautious(wallet, tower_count, min(DEFLECTOR_COST, EVASION_COST), info)
            if sale is not None:
                prefix, wallet, tower_count = sale

        can_emp = info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER))
        can_deflect = info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR))
        can_eva = info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0 and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION))

        preview = Simulator(info)
        for _ in range(16):
            if not preview.fast_next_round(1 - self.side):
                break
        base_enemy_hp = preview.info.bases[1 - self.side].hp
        base_die_count = preview.info.die_count[self.side]

        reserved_emp: List[Tuple[int, int, float]] = []
        if can_emp:
            targets = self._emp_target_positions(info)
            results: List[Tuple[int, int, float]] = []
            for x, y, _ in targets:
                if not _check_time():
                    break
                trial = Simulator(info)
                for op in prefix:
                    trial.add_operation_of_player(self.side, op)
                trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMP_BLASTER, x, y))
                trial.apply_operations_of_player(self.side)
                for _ in range(16):
                    if not trial.fast_next_round(1 - self.side):
                        break
                if self.current_round > 460:
                    if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2:
                        continue
                elif trial.info.bases[1 - self.side].hp >= base_enemy_hp - 4:
                    continue
                value = 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                for site in range(1, 34):
                    sx, sy = SITE_LAYOUT[1 - self.side][site]
                    if distance(sx, sy, x, y) <= 3:
                        bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                        value += 3 - distance(sx, sy, bx, by) * 0.01
                results.append((x, y, value))
            if results and not enemy_storm:
                x, y, _ = max(results, key=lambda item: item[2])
                self._mark_super(SuperWeaponType.EMP_BLASTER)
                return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]
            reserved_emp = results

        if can_deflect or can_eva:
            results: List[Tuple[int, int, float, bool]] = []
            if can_eva:
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        count = 0
                        min_dis = 100
                        value = 0.0
                        for ant in info.ants:
                            if ant.player == self.side and distance(ant.x, ant.y, x, y) <= 3 and ant.is_alive():
                                count += 1
                                value += ant.level + 1
                                gap = distance(ant.x, ant.y, SITE_LAYOUT[1 - self.side][HOME_SLOT][0], SITE_LAYOUT[1 - self.side][HOME_SLOT][1])
                                min_dis = min(min_dis, gap)
                        if self.current_round <= 506 and min_dis > 5:
                            continue
                        if count < 3 or (self.current_round > 460 and count < 2):
                            continue
                        trial = Simulator(info)
                        for op in prefix:
                            trial.add_operation_of_player(self.side, op)
                        trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMERGENCY_EVASION, x, y))
                        trial.apply_operations_of_player(self.side)
                        for _ in range(16):
                            if not trial.fast_next_round(1 - self.side):
                                break
                        if self.current_round > 460:
                            if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2:
                                continue
                        elif trial.info.bases[1 - self.side].hp >= base_enemy_hp - 3:
                            continue
                        value += 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                        results.append((x, y, value, True))
            if can_deflect and not results:
                bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                sx, sy = SITE_LAYOUT[1 - self.side][STORM_SLOT]
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        if distance(x, y, bx, by) > 4:
                            continue
                        trial = Simulator(info)
                        for op in prefix:
                            trial.add_operation_of_player(self.side, op)
                        trial.add_operation_of_player(self.side, Operation(OperationType.USE_DEFLECTOR, x, y))
                        trial.apply_operations_of_player(self.side)
                        for _ in range(16):
                            if not trial.fast_next_round(1 - self.side):
                                break
                        if (self.current_round > 460 and trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2) or trial.info.bases[1 - self.side].hp >= base_enemy_hp - 3:
                            continue
                        value = 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp) - distance(x, y, sx, sy)
                        results.append((x, y, value, False))
            if results:
                x, y, _, is_eva = max(results, key=lambda item: item[2])
                if is_eva:
                    self._mark_super(SuperWeaponType.EMERGENCY_EVASION)
                    return [*prefix, Operation(OperationType.USE_EMERGENCY_EVASION, x, y)]
                self._mark_super(SuperWeaponType.DEFLECTOR)
                return [*prefix, Operation(OperationType.USE_DEFLECTOR, x, y)]
        if can_emp and reserved_emp:
            x, y, _ = max(reserved_emp, key=lambda item: item[2])
            self._mark_super(SuperWeaponType.EMP_BLASTER)
            return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]
        return []

    def _try_emp(self, info: GameInfo) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] > 0:
            return []
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        enemy_wallet = info.coins[1 - self.side]
        prefix: List[Operation] = []
        if wallet - enemy_wallet < 100 or wallet < EMP_COST:
            sale = self._liquidate_cautious(wallet, tower_count, max(enemy_wallet + 100, EMP_COST), info)
            if sale is None:
                return []
            prefix, wallet, tower_count = sale
        own_preview = Simulator(info)
        for _ in range(16):
            if not own_preview.fast_next_round(self.side):
                break
            if own_preview.info.bases[self.side].hp < info.bases[self.side].hp:
                return []
        targets = self._emp_target_positions(info)
        if not targets:
            return []
        preview = Simulator(info)
        for _ in range(16):
            if not preview.fast_next_round(1 - self.side):
                break
        base_enemy_hp = preview.info.bases[1 - self.side].hp
        results: List[Tuple[int, int, float]] = []
        for x, y, base_val in targets:
            if not _check_time():
                break
            trial = Simulator(info)
            for op in prefix:
                trial.add_operation_of_player(self.side, op)
            trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMP_BLASTER, x, y))
            trial.apply_operations_of_player(self.side)
            for _ in range(16):
                if not trial.fast_next_round(1 - self.side):
                    break
            if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 4:
                continue
            value = 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
            for site in range(1, 34):
                sx, sy = SITE_LAYOUT[1 - self.side][site]
                if distance(sx, sy, x, y) <= 3:
                    bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                    value += 3 - distance(sx, sy, bx, by) * 0.01
            results.append((x, y, value))
        if not results:
            return []
        x, y, _ = max(results, key=lambda item: item[2])
        self._mark_super(SuperWeaponType.EMP_BLASTER)
        return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]

    def _try_proactive_storm(self, info: GameInfo) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)] > 0:
            return []
        if self.current_round < 25:
            return []
        cost = info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        prefix: List[Operation] = []
        can_cast = wallet >= cost
        if not can_cast:
            liq = self._liquidate_cautious(wallet, tower_count, cost, info)
            if liq is not None:
                prefix, wallet, tower_count = liq
                can_cast = wallet >= cost
        if not can_cast:
            return []
        enemy = 1 - self.side
        scored: List[Tuple[float, int, int]] = []
        seen: set = set()
        bx, by = SITE_LAYOUT[enemy][HOME_SLOT]
        for t in info.towers:
            if t.player == enemy:
                tx, ty = t.x, t.y
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        cx, cy = tx + dx, ty + dy
                        if 0 <= cx < 19 and 0 <= cy < 19 and is_valid_pos(cx, cy) and (cx, cy) not in seen:
                            seen.add((cx, cy))
                            s = 100 - distance(cx, cy, bx, by) * 5
                            scored.append((s, cx, cy))
        for ant in info.ants:
            if ant.player == enemy and ant.is_alive():
                ax, ay = ant.x, ant.y
                ad = distance(ax, ay, bx, by)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        cx, cy = ax + dx, ay + dy
                        if 0 <= cx < 19 and 0 <= cy < 19 and is_valid_pos(cx, cy) and (cx, cy) not in seen:
                            seen.add((cx, cy))
                            s = 80 - ad * 3 - distance(cx, cy, bx, by) * 3
                            scored.append((s, cx, cy))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                cx, cy = bx + dx, by + dy
                if 0 <= cx < 19 and 0 <= cy < 19 and is_valid_pos(cx, cy) and (cx, cy) not in seen:
                    seen.add((cx, cy))
                    scored.append((90 - distance(cx, cy, bx, by), cx, cy))
        sx, sy = SITE_LAYOUT[enemy][STORM_SLOT]
        if (sx, sy) not in seen:
            scored.append((80, sx, sy))
        scored.sort(key=lambda t: -t[0])
        targets = [(x, y) for _, x, y in scored[:60]]
        if not targets:
            return []
        baseline = Simulator(info)
        for op in prefix:
            baseline.add_operation_of_player(self.side, op)
        baseline.apply_operations_of_player(self.side)
        for _ in range(16):
            if not baseline.fast_next_round(self.side):
                break
        base_enemy_hp = baseline.info.bases[enemy].hp
        best_value = -1
        best_point: Optional[Tuple[int, int]] = None
        for x, y in targets:
            if not _check_time():
                break
            trial = Simulator(info)
            for op in prefix:
                trial.add_operation_of_player(self.side, op)
            trial.add_operation_of_player(self.side, Operation(OperationType.USE_LIGHTNING_STORM, x, y))
            trial.apply_operations_of_player(self.side)
            fail_round = 16
            for tick in range(16):
                if not trial.fast_next_round(self.side):
                    break
                if trial.info.bases[self.side].hp < info.bases[self.side].hp:
                    fail_round = tick
                    break
            if fail_round < 8:
                continue
            enemy_hp_delta = base_enemy_hp - trial.info.bases[enemy].hp
            if enemy_hp_delta <= 0 and self.current_round < 460:
                continue
            value = enemy_hp_delta * 100
            value += trial.info.die_count[enemy] * 5
            for t in trial.info.towers:
                if t.player == enemy and t.hp < TOWER_STATS[t.type].max_hp:
                    value += (TOWER_STATS[t.type].max_hp - t.hp) * 2
            value += fail_round
            value += (18 - distance(x, y, bx, by)) * 2
            if value > best_value:
                best_value = value
                best_point = (x, y)
        if best_point is None:
            return []
        self._mark_super(SuperWeaponType.LIGHTNING_STORM)
        return [*prefix, Operation(OperationType.USE_LIGHTNING_STORM, best_point[0], best_point[1])]

    def _try_attack(self, info: GameInfo) -> List[Operation]:
        if self.front_state == 0:
            return self._try_use_superweapon(info)
        if self.current_round <= 460:
            if info.bases[self.side].ant_level == 0:
                if info.coins[self.side] >= LEVEL2_BASE_UPGRADE_COST:
                    return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif info.bases[self.side].ant_level == 1:
                if info.coins[self.side] >= LEVEL3_BASE_UPGRADE_COST:
                    return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
                sale = self._liquidate_cautious(info.coins[self.side], info.tower_num_of_player(self.side), LEVEL3_BASE_UPGRADE_COST, info)
                if sale is not None:
                    ops, _, _ = sale
                    return [*ops, Operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif info.bases[self.side].gen_speed_level == 0:
                if info.coins[self.side] >= LEVEL2_BASE_UPGRADE_COST:
                    return [Operation(OperationType.UPGRADE_GENERATION_SPEED)]
                sale = self._liquidate_all(info.coins[self.side], info.tower_num_of_player(self.side), LEVEL2_BASE_UPGRADE_COST, info)
                if sale is not None:
                    ops, _, _ = sale
                    return [*ops, Operation(OperationType.UPGRADE_GENERATION_SPEED)]
            return self._try_use_superweapon(info)
        if self.current_round <= 470 and info.bases[self.side].ant_level == 0:
            if info.coins[self.side] >= LEVEL2_BASE_UPGRADE_COST:
                return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
            return []
        return self._try_use_superweapon(info)

    def _compute_attack(self, info: GameInfo) -> bool:
        front_state = self.front_state
        attack = front_state == -1
        own_pressure = float(info.die_count[1 - self.side])
        enemy_pressure = float(info.die_count[self.side])
        live_weight = min(1.0, (512 - self.current_round) / 20.0)
        for ant in info.ants:
            if ant.player == 1 - self.side and ant.is_alive():
                own_pressure += live_weight
            elif ant.player == self.side and ant.is_alive():
                enemy_pressure += live_weight
        if not attack and front_state == 0:
            tower_advantage = sum(1 for t in info.towers if t.player == self.side) - sum(1 for t in info.towers if t.player != self.side)
            if own_pressure - enemy_pressure >= 4:
                self.assault_memory = False
            elif own_pressure - enemy_pressure <= -3 - max((450 - self.current_round) // 50, 0):
                attack = True
            elif self.assault_memory:
                attack = True
            elif self.current_round >= 440 and own_pressure - enemy_pressure <= 2 and tower_advantage >= 0:
                attack = True
            elif self.current_round >= 460 and own_pressure - enemy_pressure <= 3:
                attack = True
        return attack

    def __call__(self, player_id: int, game_info: GameInfo) -> List[Operation]:
        global GLOBAL_TURN_START
        GLOBAL_TURN_START = time.time()

        self.current_round = game_info.round
        if self.current_round == 0:
            self.side = player_id
        enemy = 1 - self.side
        self.enemy_old_baseline = game_info.old_count[enemy]
        self.enemy_die_baseline = game_info.die_count[enemy]
        self.wall_hp_snapshot = game_info.bases[self.side].hp
        if self.current_round == 0:
            self.initial_enemy_tower_count = sum(1 for t in game_info.towers if t.player == enemy)
        my_hp = game_info.bases[self.side].hp
        enemy_hp = game_info.bases[enemy].hp
        self.front_state = 1 if my_hp > enemy_hp else (-1 if my_hp < enemy_hp else 0)
        attack = self._compute_attack(game_info)

        enemy_emp = -1
        for weapon in game_info.super_weapons:
            if weapon.player == enemy and weapon.type == SuperWeaponType.EMP_BLASTER:
                enemy_emp = weapon.left_time
                break

        if _check_time():
            ops = self._try_proactive_storm(game_info)
            if ops:
                return ops

        if self.front_state <= 0 and _check_time():
            ops = self._try_emp(game_info)
            if ops:
                return ops
        if not _check_time():
            return []

        if attack and not self.reserve_depth:
            self.assault_memory = True
            ops = self._try_attack(game_info)
            if ops:
                return ops
        if not _check_time():
            return []

        if not self.reserve_depth and _check_time():
            ops = self._try_use_superweapon(game_info)
            if ops:
                return ops

        if self.front_state == 1 and self.current_round >= 488:
            ops = self._try_end_storm(game_info)
            if ops:
                return ops
        if self.front_state == 0 and self.current_round >= 510:
            ops = self._try_use_storm(game_info, True)
            if ops:
                return ops
        if not _check_time():
            return []

        staging = game_info.clone()
        staging.bases[enemy].hp = max(SEARCH_STAGING_ENEMY_BASE_HP, game_info.bases[enemy].hp)

        self.nodes = []
        root = ForecastNode(self, Simulator(staging))
        root.node_id = 0
        root.parent = -1
        root.evaluate()
        self.nodes.append(root)
        self.nodes[0].expand(is_root=True)

        search_start = time.process_time()
        while True:
            if time.process_time() - search_start >= SEARCH_BUDGET:
                break
            if len(self.nodes) >= MAX_NODE_COUNT - 10:
                break
            if not self._expand_one():
                break

        if _check_time():
            self._support_expand(4)

        best_id = -1
        best_value = -1e9
        for child_id in self.nodes[0].children:
            child = self.nodes[child_id]
            if child.best_descendant > best_value:
                best_value = child.best_descendant
                best_id = child_id

        if len(self.nodes) > 1 and best_id > 1 and best_value - self.nodes[1].best_descendant < 2:
            best_id = 1
            best_value = self.nodes[1].best_descendant

        if best_id >= 0:
            imminent = self.nodes[best_id].collapse_round - self.current_round
            emergency_storm = (
                (
                    self.front_state >= 0
                    and (
                        (enemy_emp > 0 and imminent < min(8, enemy_emp) and best_value < -400)
                        or (best_value < -700 and imminent <= 2)
                    )
                )
                or (
                    self.front_state == 0
                    and game_info.die_count[enemy] - game_info.die_count[self.side] >= 8
                    and imminent <= 1
                )
            )
            if emergency_storm:
                ops = self._try_use_storm(game_info, self.current_round >= 480)
                if ops:
                    return ops

        if best_id > 0:
            self.reserve_depth = self.nodes[best_id].best_depth
            return list(self.nodes[best_id].chosen)
        return []
