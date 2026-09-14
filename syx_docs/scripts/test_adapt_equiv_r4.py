"""Round-4 adapt-side optimization equivalence tests.

Compares the vectorized implementations (map_process / agent_process, rewritten
2026-08-20) against the original loop-based reference implementations embedded
below (verbatim from git HEAD before the rewrite). Run inside the server conda
env (needs nuplan-devkit imports):

    python test_adapt_equiv_r4.py
"""
import numpy as np

from diffusion_planner.data_process.map_process import (
    _interpolate_points,
    _interpolate_points_batch,
    _lane_polyline_process,
)
from diffusion_planner.data_process.agent_process import (
    _filter_agents_array,
    _pad_agent_states,
    _extract_agent_array,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

rng = np.random.default_rng(20260820)


# ===================== reference (old) implementations =====================
def _old_lane_polyline_process(polylines, left_boundary, right_boundary, avails, traffic_light):
    dim = 12
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_vector = polyline[1:] - polyline[:-1]
            polyline_vector = np.insert(polyline_vector, polyline_vector.shape[0], 0, axis=0)

            if np.linalg.norm(left_boundary[i, -1] - polyline[0]) < np.linalg.norm(left_boundary[i, 0] - polyline[0]):
                left_boundary[i] = np.flip(left_boundary[i], axis=0)

            if np.linalg.norm(right_boundary[i, -1] - polyline[0]) < np.linalg.norm(right_boundary[i, 0] - polyline[0]):
                right_boundary[i] = np.flip(right_boundary[i], axis=0)

            polyline_to_left = left_boundary[i] - polyline
            polyline_to_right = right_boundary[i] - polyline

            new_polylines[i] = np.concatenate([polyline, polyline_vector, polyline_to_left, polyline_to_right, traffic_light[i]], axis=-1)

    return new_polylines


def _old_filter_agents_array(agents, reverse: bool = False):
    target_array = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):
        rows = []
        for j in range(agents[i].shape[0]):
            if target_array.shape[0] > 0:
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())])
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:, AgentInternalIndex.track_token()]).max()
                )
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())
        if len(rows) > 0:
            agents[i] = np.stack(rows)
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)
    return agents


def _old_pad_agent_states(agent_trajectories, reverse: bool):
    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]
    id_row_mapping = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val)] = idx

    current_state = np.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=np.float64)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        for row_idx in range(frame.shape[0]):
            mapped_row = id_row_mapping[int(frame[row_idx, track_id_idx])]
            current_state[mapped_row, :] = frame[row_idx, :]
        agent_trajectories[idx] = current_state.copy()

    if reverse:
        agent_trajectories = agent_trajectories[::-1]
    return agent_trajectories


def _old_extract_agent_array(tracked_objects, track_token_ids, object_types):
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = np.zeros((len(agents), AgentInternalIndex.dim()), dtype=np.float64)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]
        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


# ===================== mocks =====================
class _Box:
    def __init__(self, w, l):
        self.width, self.length = w, l


class _Center:
    def __init__(self, x, y, h):
        self.x, self.y, self.heading = x, y, h


class _Vel:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Agent:
    def __init__(self, token, x, y, h, vx, vy, w, l):
        self.track_token, self.center, self.velocity, self.box = token, _Center(x, y, h), _Vel(vx, vy), _Box(w, l)
        self.tracked_object_type = TrackedObjectType.VEHICLE


class _TrackedObjects:
    def __init__(self, agents):
        self._agents = agents

    def get_tracked_objects_of_types(self, types):
        return self._agents


# ===================== test cases =====================
def test_batch_interpolate(n_cases=900):
    worst = 0.0
    for _ in range(n_cases):
        n_lines = int(rng.integers(0, 40))
        lines = []
        for _ in range(n_lines):
            n_pts = int(rng.integers(1, 60))
            if rng.random() < 0.1:
                pt = rng.normal(size=2)
                arr = np.repeat(pt[None], n_pts, axis=0)  # zero-length line
            elif rng.random() < 0.05:
                arr = rng.normal(size=(1, 2))  # single point
            else:
                arr = np.cumsum(rng.normal(size=(n_pts, 2)) * 0.5, axis=0)
            lines.append(arr)
        num_point = int(rng.integers(2, 25))

        batch = _interpolate_points_batch(lines, num_point)
        ref = (
            np.zeros((n_lines, num_point, 2))
            if n_lines == 0
            else np.stack([_interpolate_points(l, num_point) for l in lines])
        )
        if batch.shape != ref.shape:
            raise AssertionError(f"shape {batch.shape} != {ref.shape}")
        diff = np.max(np.abs(batch - ref)) if ref.size else 0.0
        worst = max(worst, diff)
    print(f"batch_interpolate: {n_cases} cases, worst |diff| = {worst:.3e}")
    assert worst < 1e-10


def test_lane_polyline_process(n_cases=300):
    worst = 0.0
    for _ in range(n_cases):
        E = int(rng.integers(1, 100))
        P = int(rng.integers(2, 25))
        polylines = rng.normal(size=(E, P, 2)).astype(np.float32)
        left = rng.normal(size=(E, P, 2)).astype(np.float32)
        right = rng.normal(size=(E, P, 2)).astype(np.float32)
        tl = (rng.random((E, P, 4)) > 0.5).astype(np.float32)
        avails = np.zeros((E, P), dtype=np.bool_)
        n_valid = int(rng.integers(0, E + 1))
        valid_rows = rng.choice(E, size=n_valid, replace=False)
        avails[valid_rows] = True
        for arr in (polylines, left, right, tl):
            arr[~avails] = 0.0

        new = _lane_polyline_process(polylines.copy(), left.copy(), right.copy(), avails, tl.copy())
        ref = _old_lane_polyline_process(polylines.copy(), left.copy(), right.copy(), avails, tl.copy())
        diff = np.max(np.abs(new - ref)) if ref.size else 0.0
        worst = max(worst, diff)
    print(f"lane_polyline_process: {n_cases} cases, worst |diff| = {worst:.3e}")
    assert worst == 0.0


def _random_frames(n_frames=11, max_agents=40):
    """Build a list of frames [N_i, dim] with shared track tokens, mimicking
    sampled_tracked_objects_to_array_list output."""
    dim = AgentInternalIndex.dim()
    n_ids = int(rng.integers(1, max_agents))
    frames = []
    for _ in range(n_frames):
        present = rng.random(n_ids) < 0.8
        ids = np.flatnonzero(present).astype(np.float64)
        if rng.random() < 0.05:
            ids = ids[:0]  # occasionally an empty frame
        frame = np.zeros((ids.shape[0], dim))
        frame[:, int(AgentInternalIndex.track_token())] = ids
        frame[:, :int(AgentInternalIndex.track_token())] = rng.normal(size=(ids.shape[0], int(AgentInternalIndex.track_token())))
        frames.append(frame)
    return frames


def test_filter_and_pad(n_cases=300):
    worst = 0.0
    for _ in range(n_cases):
        frames = _random_frames()
        reverse = bool(rng.random() < 0.5)

        got = [f.copy() for f in frames]
        got = _filter_agents_array(got, reverse=reverse)
        got = _pad_agent_states(got, reverse=reverse)

        ref = [f.copy() for f in frames]
        ref = _old_filter_agents_array(ref, reverse=reverse)
        if any(f.shape[0] == 0 for f in ref):
            continue  # zero-agent path short-circuits in agent_past_process
        ref = _old_pad_agent_states(ref, reverse=reverse)

        for g, r in zip(got, ref):
            if g.shape != r.shape or not np.array_equal(g, r):
                worst = max(worst, np.max(np.abs(g - r)) if g.size and g.shape == r.shape else 1.0)
    print(f"filter+pad_agents: {n_cases} cases, worst |diff| = {worst:.3e}")
    assert worst == 0.0


def test_extract_agent_array(n_cases=200):
    worst = 0.0
    for _ in range(n_cases):
        n_agents = int(rng.integers(0, 60))
        tokens = [f"tok{i}" for i in rng.permutation(200)[:n_agents]]
        agents = [
            _Agent(t, *rng.normal(size=7))
            for t in tokens
        ]
        tracked = _TrackedObjects(agents)
        types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]

        seed_ids = {}
        for _ in range(int(rng.integers(0, 5))):  # pre-populated token dict
            seed_ids[f"pre{len(seed_ids)}"] = len(seed_ids)

        got_out, got_ids, _ = _extract_agent_array(tracked, dict(seed_ids), types)
        ref_out, ref_ids, _ = _old_extract_agent_array(tracked, dict(seed_ids), types)

        if got_out.shape != ref_out.shape or not np.array_equal(got_out, ref_out) or got_ids != ref_ids:
            worst = max(worst, np.max(np.abs(got_out - ref_out)) if got_out.size else 1.0)
    print(f"extract_agent_array: {n_cases} cases, worst |diff| = {worst:.3e}")
    assert worst == 0.0


if __name__ == "__main__":
    test_batch_interpolate()
    test_lane_polyline_process()
    test_filter_and_pad()
    test_extract_agent_array()
    print("ALL_EQUIV_OK")
