from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np


ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
SIM_ACTION_NAMES = {1: "move_forward", 2: "turn_left", 3: "turn_right"}
ACTION_UNITS = {1: "m", 2: "deg", 3: "deg"}
NOISE_DISTRIBUTION = "symmetric_uniform_relative_v1"


@dataclass(frozen=True)
class ActionExecutionPlan:
    expert_action: int
    nominal_amount: Optional[float]
    sampled_amount: Optional[float]
    scale: Optional[float]
    unit: Optional[str]


@dataclass(frozen=True)
class ActionExecutionResult:
    observation: Any
    executed: bool
    achieved_translation_m: Optional[float]
    achieved_yaw_deg: Optional[float]
    collision: Optional[bool]


class ActionExecutionError(RuntimeError):
    def __init__(self, message, executed=False, result=None):
        super().__init__(message)
        self.executed = bool(executed)
        self.result = result


def stable_relative_scale(
    seed: int,
    scene_id: str,
    task_id: str,
    step: int,
    action: int,
    factor: float,
) -> float:
    factor = float(factor)
    if factor == 0.0:
        return 1.0
    payload = "\0".join(
        (str(int(seed)), str(scene_id), str(task_id), str(int(step)), str(int(action)))
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    unit = value / float((1 << 64) - 1)
    return 1.0 + factor * (2.0 * unit - 1.0)


def action_debug_record(
    plan: ActionExecutionPlan,
    result: Optional[ActionExecutionResult] = None,
) -> Dict[str, Any]:
    return {
        "expert_action": int(plan.expert_action),
        "action_name": ACTION_NAMES[int(plan.expert_action)],
        "executed": bool(result.executed) if result is not None else False,
        "nominal_amount": plan.nominal_amount,
        "sampled_amount": plan.sampled_amount,
        "unit": plan.unit,
        "scale": plan.scale,
        "achieved_translation_m": (
            result.achieved_translation_m if result is not None else None
        ),
        "achieved_yaw_deg": (
            result.achieved_yaw_deg if result is not None else None
        ),
        "collision": result.collision if result is not None else None,
    }


class PerturbedActionExecutor:
    def __init__(self, sim, factor: float = 0.0, seed: int = 42,
                 verbose: bool = False):
        self.sim = sim
        self.factor = float(factor)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        if not np.isfinite(self.factor) or not 0.0 <= self.factor < 1.0:
            raise ValueError("factor must satisfy 0 <= factor < 1")
        if self.seed < 0 or self.seed >= (1 << 63):
            raise ValueError("seed must be in [0, 2^63)")
        self._specs: Dict[int, Any] = {}
        self._nominal: Dict[int, float] = {}
        if self.factor > 0.0 or self.verbose:
            self._bind_action_specs()

    def _candidate_agents(self):
        seen = set()
        owners = [self.sim, getattr(self.sim, "_sim", None)]
        agent_ids = [
            getattr(self.sim, "_default_agent_id", None),
            getattr(self.sim, "default_agent_id", None),
            0,
        ]
        for owner in owners:
            if owner is None:
                continue
            getter = getattr(owner, "get_agent", None)
            if callable(getter):
                for agent_id in agent_ids:
                    if agent_id is None:
                        continue
                    try:
                        agent = getter(int(agent_id))
                    except Exception:
                        continue
                    if id(agent) not in seen:
                        seen.add(id(agent))
                        yield agent
            for agent in getattr(owner, "agents", []) or []:
                if id(agent) not in seen:
                    seen.add(id(agent))
                    yield agent

    @staticmethod
    def _action_space(agent):
        for config_name in ("agent_config", "config", "configuration"):
            config = getattr(agent, config_name, None)
            action_space = getattr(config, "action_space", None)
            if action_space is not None:
                return action_space
        return None

    @staticmethod
    def _spec_name(key, spec) -> str:
        values = [key, getattr(spec, "name", None)]
        return " ".join(str(value).lower() for value in values if value is not None)

    def _bind_action_specs(self):
        spaces = [self._action_space(agent) for agent in self._candidate_agents()]
        spaces = [space for space in spaces if hasattr(space, "items")]
        for action, expected_name in SIM_ACTION_NAMES.items():
            matches = []
            for action_space in spaces:
                for key, spec in action_space.items():
                    if expected_name in self._spec_name(key, spec):
                        actuation = getattr(spec, "actuation", None)
                        if hasattr(actuation, "amount"):
                            matches.append(spec)
            if not matches:
                raise RuntimeError(
                    f"Cannot resolve writable Habitat action spec for {expected_name}; "
                    "actuation noise requires a supported Habitat-Sim action space"
                )
            spec = matches[0]
            amount = float(spec.actuation.amount)
            if not np.isfinite(amount) or amount <= 0.0:
                raise RuntimeError(
                    f"Invalid nominal Habitat actuation amount for {expected_name}: "
                    f"{amount!r}"
                )
            spec.actuation.amount = amount
            if float(spec.actuation.amount) != amount:
                raise RuntimeError(f"Habitat actuation amount is not writable: {expected_name}")
            self._specs[action] = spec
            self._nominal[action] = amount

    def plan(self, action: int, scene_id: str, task_id: str,
             step: int) -> ActionExecutionPlan:
        action = int(action)
        if action == 0:
            return ActionExecutionPlan(0, None, None, None, None)
        if action not in SIM_ACTION_NAMES:
            raise ValueError(f"Unsupported navigation action: {action}")
        if not self._specs:
            return ActionExecutionPlan(action, None, None, 1.0, ACTION_UNITS[action])
        nominal = self._nominal[action]
        scale = stable_relative_scale(
            self.seed, scene_id, task_id, step, action, self.factor
        )
        return ActionExecutionPlan(
            action, nominal, nominal * scale, scale, ACTION_UNITS[action]
        )

    @staticmethod
    def _copy_position(state) -> Optional[np.ndarray]:
        try:
            return np.asarray(state.position, dtype=float).copy()
        except Exception:
            return None

    @staticmethod
    def _heading(state) -> Optional[float]:
        try:
            from habitat.utils.geometry_utils import quaternion_rotate_vector

            forward = quaternion_rotate_vector(
                state.rotation, np.asarray([0.0, 0.0, -1.0])
            )
            return float(np.arctan2(float(forward[0]), -float(forward[2])))
        except Exception:
            return None

    def _collision(self) -> Optional[bool]:
        for owner in (self.sim, getattr(self.sim, "_sim", None)):
            value = getattr(owner, "previous_step_collided", None)
            if value is not None:
                try:
                    return bool(value() if callable(value) else value)
                except Exception:
                    pass
        return None

    def _measure_result(self, observation, pre_position, pre_heading):
        translation = yaw = None
        if self.verbose:
            try:
                post_state = self.sim.get_agent_state()
                post_position = self._copy_position(post_state)
                post_heading = self._heading(post_state)
                if pre_position is not None and post_position is not None:
                    translation = float(
                        np.linalg.norm(post_position - pre_position)
                    )
                if pre_heading is not None and post_heading is not None:
                    delta = (
                        post_heading - pre_heading + np.pi
                    ) % (2 * np.pi) - np.pi
                    yaw = float(np.degrees(delta))
            except Exception:
                # Diagnostics must never turn a successfully executed action
                # into an unsaved/ambiguous transition.
                translation = yaw = None
        return ActionExecutionResult(
            observation, True, translation, yaw,
            self._collision() if self.verbose else None,
        )

    def execute(self, env, plan: ActionExecutionPlan) -> ActionExecutionResult:
        if plan.expert_action == 0:
            raise ValueError("STOP must not be passed to the actuation executor")
        try:
            pre_state = self.sim.get_agent_state() if self.verbose else None
        except Exception:
            pre_state = None
        pre_position = self._copy_position(pre_state) if pre_state is not None else None
        pre_heading = self._heading(pre_state) if pre_state is not None else None

        # factor=0/verbose=false is an exact fast path through the old env.step.
        if plan.nominal_amount is None:
            try:
                observation = env.step({"action": int(plan.expert_action)})
            except Exception as error:
                raise ActionExecutionError(str(error), executed=False) from error
            return self._measure_result(observation, pre_position, pre_heading)

        spec = self._specs[int(plan.expert_action)]
        observation = None
        step_error = restore_error = None
        try:
            spec.actuation.amount = float(plan.sampled_amount)
            observation = env.step({"action": int(plan.expert_action)})
        except Exception as error:
            step_error = error
        finally:
            try:
                spec.actuation.amount = float(plan.nominal_amount)
            except Exception as error:
                restore_error = error

        if step_error is not None:
            raise ActionExecutionError(str(step_error), executed=False) from step_error
        result = self._measure_result(observation, pre_position, pre_heading)
        if restore_error is not None:
            raise ActionExecutionError(
                f"Action executed but nominal actuation restore failed: {restore_error}",
                executed=True,
                result=result,
            ) from restore_error
        return result

