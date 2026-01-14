import os
import torch
import math
import pickle
import numpy as np
from torch import nn
from typing import Dict, Sequence
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from utils.misc import wrap_angle
from pathlib import Path

SHIFT = 5
AGENT_SHAPE = {
    "vehicle": [4.3, 1.8, 1.0],
    "pedstrain": [0.5, 0.5, 1.0],
    "cyclist": [1.9, 0.5, 1.0],
}
AGENT_TYPE = ["veh", "ped", "cyc", "seed"]
AGENT_STATE = ["invalid", "valid", "enter", "exit"]


@torch.no_grad()
def cal_polygon_contour(
    pos, head, width_length
) -> torch.Tensor:  # [n_agent, n_step, n_target, 4, 2]
    x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
    width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1, 1]

    half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
    half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
    length_cos = length * half_cos  # [n_agent, n_step, n_target]
    length_sin = length * half_sin  # [n_agent, n_step, n_target]
    width_cos = width * half_cos  # [n_agent, n_step, n_target]
    width_sin = width * half_sin  # [n_agent, n_step, n_target]

    left_front_x = x + length_cos - width_sin
    left_front_y = y + length_sin + width_cos
    left_front = torch.stack((left_front_x, left_front_y), dim=-1)

    right_front_x = x + length_cos + width_sin
    right_front_y = y + length_sin - width_cos
    right_front = torch.stack((right_front_x, right_front_y), dim=-1)

    right_back_x = x - length_cos + width_sin
    right_back_y = y - length_sin - width_cos
    right_back = torch.stack((right_back_x, right_back_y), dim=-1)

    left_back_x = x - length_cos - width_sin
    left_back_y = y - length_sin + width_cos
    left_back = torch.stack((left_back_x, left_back_y), dim=-1)

    polygon_contour = torch.stack(
        (left_front, right_front, right_back, left_back), dim=-2
    )

    return polygon_contour


def interplating_polyline(polylines, heading, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(
            abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
            abs(
                max(heading[i], heading[i - 1])
                - min(heading[1], heading[i - 1])
                + math.pi
            ),
        )
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(
                dist_along_path_list[-1][-1] + euclidean_dist
            )
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate(
            [new_dist_along_path, dist_along_path[[-1]]]
        )
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) % polyline_size
            final_index = (
                new_polylines.shape[0] - (polyline_size + 1)
            ) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0],
        )
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(
                dimension=0, size=polyline_size + 1, step=polyline_size
            )
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size :]
            last_polyline = last_polyline[
                torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()
            ]
            if multi_polylines is not None:
                multi_polylines = torch.cat(
                    [multi_polylines, last_polyline.unsqueeze(0)], dim=0
                )
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def average_distance_vectorized(point_set1, centroids):
    dists = np.sqrt(
        np.sum((point_set1[:, None, :, :] - centroids[None, :, :, :]) ** 2, axis=-1)
    )
    return np.mean(dists, axis=2)


def assign_clusters(sub_X, centroids):
    distances = average_distance_vectorized(sub_X, centroids)
    return np.argmin(distances, axis=1)


class TokenProcessor(nn.Module):

    def __init__(
        self,
        token_size,
        training: bool = False,
        predict_motion: bool = False,
        predict_state: bool = False,
        predict_map: bool = False,
        state_token: Dict[str, int] = None,
        **kwargs,
    ):
        """_summary_

        Args:
            token_size (_type_): _description_
            training (bool, optional): _description_. Defaults to False.
            predict_motion (bool, optional): _description_. Defaults to False.
            predict_state (bool, optional): _description_. Defaults to False.
            predict_map (bool, optional): _description_. Defaults to False.
            state_token: {'invalid': 0, 'valid': 1, 'enter': 2, 'exit': 3}
        Raises:
            FileNotFoundError: _description_
        """
        super().__init__()

        self.agent_token_path = Path(
            "/home/tjhu78u/workspace/motion_prediction/models/infgen/agent_vocab_555_s2.pkl"
        )
        self.map_token_traj_path = Path(
            "/home/tjhu78u/workspace/motion_prediction/models/infgen/map_traj_token5.pkl"
        )

        if not self.agent_token_path.exists() or not self.map_token_traj_path.exists():
            raise FileNotFoundError(
                "Token files not found. Please ensure the paths are correct."
            )

        self.training = training
        self.token_size = token_size
        self.disable_invalid = not predict_state
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map

        # define new special tokens
        self.bos_token_index = token_size
        self.eos_token_index = token_size + 1
        self.invalid_token_index = token_size + 2
        self.special_token_index = []
        self._init_token()

        # define agent states
        self.invalid_state = int(state_token["invalid"])
        self.valid_state = int(state_token["valid"])
        self.enter_state = int(state_token["enter"])
        self.exit_state = int(state_token["exit"])

        self.pl2seed_radius = kwargs.get("pl2seed_radius", None)  # 75

        self.noise = False
        self.disturb = False
        self.shift = 5
        self.training = False
        self.current_step = 10

        # debugging
        self.debug_data = None

    def forward(self, data):
        """
        Each pkl data represents a extracted scenario from raw tfrecord data
        """
        data["agent"]["av_index"] = data["agent"]["av_idx"]
        data = self._tokenize_agent(data)
        # data = self._tokenize_map(data)
        del data["city"]
        if "polygon_is_intersection" in data["map_polygon"]:
            del data["map_polygon"]["polygon_is_intersection"]
        if "route_type" in data["map_polygon"]:
            del data["map_polygon"]["route_type"]

        av_index = data["agent"]["av_idx"]
        data["ego_pos"] = data["agent"]["token_pos"][[av_index]]
        data["ego_heading"] = data["agent"]["token_heading"][[av_index]]

        return data

    def _init_token(self):
        """register motion vocab
        - agent_token_all_vec: (2048, 6, 4, 2): start from veh shape (4.8, 2)
        - agent_token_all_ped: (2048, 6, 4, 2)
        - agent_token_all_cyc: (2048, 6, 4, 2)
        - map_token_traj_src: (1024, 11, 2): start from (0, 0)
        """

        agent_token_data = pickle.load(open(self.agent_token_path, "rb"))
        for agent_type, token in agent_token_data["token_all"].items():
            token = torch.tensor(token, dtype=torch.float32)
            self.register_buffer(
                f"agent_token_all_{agent_type}", token, persistent=False
            )  # [n_token, 6, 4, 2]

        map_token_traj = pickle.load(open(self.map_token_traj_path, "rb"))["traj_src"]
        map_token_traj = torch.tensor(map_token_traj, dtype=torch.float32)
        self.register_buffer(
            "map_token_traj_src", map_token_traj, persistent=False
        )  # [n_token, 11 * 2]

    @staticmethod
    def clean_heading(
        valid: torch.Tensor,
        heading: torch.Tensor,
        thres: float = 1.5,
    ) -> torch.Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > thres) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading

    def _extrapolate_agent_to_prev_token_step(
        self, valid, pos, heading, vel
    ) -> Sequence[torch.Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices  # [n_agent]

        for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
            n_step_to_extrapolate = t % self.shift
            if (t == self.current_step) and (
                not valid[i, self.current_step - self.shift]
            ):
                # such that at least one token is valid in the history.
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

                for j in range(n_step_to_extrapolate):
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

        return valid, pos, heading, vel

    def _get_agent_shape(self, agent_type_masks: dict) -> torch.Tensor:
        agent_shape = 0.0
        _AGENT_SHAPE = {
            "veh": (4.8, 2.0),
            "ped": (2.0, 1.0),
            "cyc": (1.0, 1.0),
        }
        for type, type_mask in agent_type_masks.items():
            length, width = _AGENT_SHAPE[type]
            if type == "veh":
                width = 2.0
                length = 4.8
            if type == "ped":
                width = 1.0
                length = 2.0
            if type == "cyc":
                width = 1.0
                length = 1.0
            agent_shape += torch.stack([width * type_mask, length * type_mask], dim=-1)

        return agent_shape

    def _get_token_traj_all(self, agent_type_masks: dict) -> torch.Tensor:
        token_traj_all = 0.0
        for type, type_mask in agent_type_masks.items():
            token_traj_all += type_mask[:, None, None, None, None] * (
                getattr(self, f"agent_token_all_{type}").unsqueeze(0)
            )
        return token_traj_all

    def _tokenize_agent(self, data): 
        # get raw data
        valid_mask = data["agent"]["valid_mask"]  # [n_agent, n_step]
        agent_heading = data["agent"]["heading"]  # [n_agent, n_step]
        agent_pos = data["agent"]["position"][
            ..., :2
        ].contiguous()  # [n_agent, n_step, 2]
        agent_vel = data["agent"]["velocity"]  # [n_agent, n_step, 2]
        agent_type = data["agent"]["type"]
        agent_category = data["agent"]["category"]

        n_agent, n_all_step = valid_mask.shape

        agent_type_masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        agent_heading = self.clean_heading(valid_mask, agent_heading)
        agent_shape = self._get_agent_shape(agent_type_masks)
        token_traj_all = self._get_token_traj_all(agent_type_masks)  # (n_a, 2048, 6, 4, 2), each agent's vocab
        valid_mask, agent_pos, agent_heading, agent_vel = (
            self._extrapolate_agent_to_prev_token_step(
                valid_mask, agent_pos, agent_heading, agent_vel
            )
        )
        token_traj = token_traj_all[:, :, -1, ...]  # (n_a, 2048, 4, 2), each agent's vocab last contour
        data["agent"]["token_traj_all"] = token_traj_all  # [n_agent, n_token, 6, 4, 2]
        data["agent"]["token_traj"] = token_traj  # [n_agent, n_token, 4, 2]

        valid_mask_shift = valid_mask.unfold(1, self.shift + 1, self.shift)
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]

        # index: [n_agent, n_step] contour: [n_agent, n_step, 4, 2]
        token_index, token_contour, token_all = self._match_agent_token(
            valid_mask,
            agent_pos,
            agent_heading,
            agent_shape,
            token_traj,
            None,  # token_traj_all
        )

        traj_pos = traj_heading = None
        if len(token_all) > 0:
            traj_pos = token_all.mean(dim=3)  # [n_agent, n_step, 6, 2]
            diff_xy = token_all[..., 0, :] - token_all[..., 3, :]
            traj_heading = torch.arctan2(diff_xy[..., 1], diff_xy[..., 0])
        token_pos = token_contour.mean(dim=2)  # [n_agent, n_step, 2]
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        token_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        # ! compute agent states
        bos_index = torch.argmax(token_valid_mask.long(), dim=1)
        eos_index = (
            token_valid_mask.shape[1]
            - 1
            - torch.argmax(torch.flip(token_valid_mask.long(), dims=[1]), dim=1)
        )
        state_index = torch.ones_like(token_index)  # init with all valid
        step_index = (
            torch.arange(state_index.shape[1])[None]
            .repeat(state_index.shape[0], 1)
            .to(token_index.device)
        )
        state_index[step_index == bos_index[:, None]] = self.enter_state
        state_index[step_index == eos_index[:, None]] = self.exit_state
        state_index[
            (step_index < bos_index[:, None]) | (step_index > eos_index[:, None])
        ] = self.invalid_state
        # ! IMPORTANT: if the last step is exit token, should convert it back to valid token
        state_index[state_index[:, -1] == self.exit_state, -1] = self.valid_state

        # update token attributions according to state tokens
        token_valid_mask[state_index == self.enter_state] = False
        token_pos[state_index == self.invalid_state] = 0.0
        token_heading[state_index == self.invalid_state] = 0.0
        for i in range(self.shift, agent_pos.shape[1], self.shift):
            is_bos = state_index[:, i // self.shift - 1] == self.enter_state
            token_pos[is_bos, i // self.shift - 1] = agent_pos[is_bos, i].clone()
            # token_heading[is_bos, i // self.shift - 1] = agent_heading[is_bos, i].clone()
        token_index[state_index == self.invalid_state] = -1
        token_index[state_index == self.enter_state] = -2

        raw_token_valid_mask = token_valid_mask.clone()
        if not self.disable_invalid:
            token_valid_mask = torch.ones_like(token_valid_mask).bool()

        # reset agent shapes
        for i in range(n_agent):
            bos_shape_index = torch.nonzero(
                torch.all(data["agent"]["shape"][i] != 0.0, dim=-1)
            )[0]
            data["agent"]["shape"][i, :] = data["agent"]["shape"][i, bos_shape_index]
        if torch.any(torch.all(data["agent"]["shape"][i] == 0.0, dim=-1)):
            raise ValueError(f"Found invalid shape values.")

        # compute mean height values for each scenario
        raw_height = data["agent"]["position"][:, self.current_step, 2]
        valid_height = raw_token_valid_mask[:, 1].bool()
        veh_mean_z = raw_height[agent_type_masks["veh"] & valid_height].mean()
        ped_mean_z = (
            raw_height[agent_type_masks["ped"] & valid_height]
            .mean()
            .nan_to_num_(veh_mean_z)
        )  # FIXME: hard code
        cyc_mean_z = (
            raw_height[agent_type_masks["cyc"] & valid_height]
            .mean()
            .nan_to_num_(veh_mean_z)
        )

        # output
        data["agent"]["token_idx"] = token_index
        data["agent"]["state_idx"] = state_index
        data["agent"]["token_contour"] = token_contour
        data["agent"]["traj_pos"] = traj_pos
        data["agent"]["traj_heading"] = traj_heading
        data["agent"]["token_pos"] = token_pos
        data["agent"]["token_heading"] = token_heading
        data["agent"]["agent_valid_mask"] = token_valid_mask  # (a, t)
        data["agent"]["raw_agent_valid_mask"] = raw_token_valid_mask
        data["agent"]["raw_height"] = dict(
            veh=veh_mean_z, ped=ped_mean_z, cyc=cyc_mean_z
        )
        for type in ["veh", "ped", "cyc"]:
            data["agent"][f"trajectory_token_{type}"] = getattr(
                self, f"agent_token_all_{type}"
            )  # [n_token, 6, 4, 2]

        return data

    def _match_agent_token(
        self, valid_mask, pos, heading, shape, token_traj, token_traj_all=None
    ):
        """
        Parameters:
        valid_mask (torch.Tensor): Validity mask for agents over time. Shape: (n_agent, n_step)
        pos (torch.Tensor): Positions of agents at each time step. Shape: (n_agent, n_step, 3)
        heading (torch.Tensor): Headings of agents at each time step. Shape: (n_agent, n_step)
        shape (torch.Tensor): Shape information of agents. Shape: (n_agent, 3)
        token_traj (torch.Tensor): Token trajectories for agents. Shape: (n_agent, n_token, 4, 2)
        token_traj_all (torch.Tensor): Token trajectories for all agents. Shape: (n_agnet, n_token_all, n_contour, 4, 2)

        Returns:
        tuple: Contains token indices and contours for agents.
        """

        n_agent, n_step = valid_mask.shape

        _, n_token, token_contour_dim, feat_dim = token_traj.shape
        # agent_token_src = agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim).repeat(agent_num, 0)

        token_index_list = []
        token_contour_list = []
        token_all = []

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        prev_token_idx = None
        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid_mask[:, i - self.shift] & valid_mask[:, i]
            _invalid_mask = ~_valid_mask

            # transformation
            theta = prev_heading
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(n_agent, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(token_traj.flatten(1, 2), rot_mat).reshape(
                *token_traj.shape
            )
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(
                pos[:, i], heading[:, i], shape
            )  # [n_agent, 4, 2]
            agent_token_index = torch.argmin(
                torch.norm(agent_token_world - cur_contour[:, None, ...], dim=-1).sum(
                    -1
                ),
                dim=-1,
            )
            agent_token_contour = agent_token_world[
                torch.arange(n_agent), agent_token_index
            ]  # [n_agent, 4, 2]
            # except for the first timestep TODO
            if prev_token_idx is not None and self.noise:
                same_idx = prev_token_idx == agent_token_index
                same_idx[:] = True
                topk_indices = np.argsort(
                    np.mean(
                        np.sqrt(
                            np.sum(
                                (cur_contour[:, None, ...] - agent_token_world.numpy())
                                ** 2,
                                axis=-1,
                            )
                        ),
                        axis=2,
                    ),
                    axis=-1,
                )[:, :5]
                sample_topk = np.random.choice(
                    range(0, topk_indices.shape[1]), topk_indices.shape[0]
                )
                agent_token_index[same_idx] = torch.from_numpy(
                    topk_indices[np.arange(topk_indices.shape[0]), sample_topk]
                )[same_idx]

            # update prev_heading
            prev_heading = heading[:, i].clone()
            diff_xy = agent_token_contour[:, 0] - agent_token_contour[:, 3]
            prev_heading[_valid_mask] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])[
                _valid_mask
            ]

            # update prev_pos
            prev_pos = pos[:, i].clone()
            prev_pos[_valid_mask] = agent_token_contour.mean(dim=1)[_valid_mask]

            prev_token_idx = agent_token_index
            token_index_list.append(agent_token_index)
            token_contour_list.append(agent_token_contour)

            # calculate tokenized trajectory
            if token_traj_all is not None:
                agent_token_all_world = torch.bmm(
                    token_traj_all.flatten(1, 3), rot_mat
                ).reshape(*token_traj_all.shape)
                agent_token_all_world += prev_pos[:, None, None, None, :]
                agent_token_all = agent_token_all_world[
                    torch.arange(n_agent), agent_token_index
                ]  # [n_agent, 6, 4, 2]
                token_all.append(agent_token_all)

        token_index = torch.stack(token_index_list, dim=1)  # [n_agent, n_step]
        token_contour = torch.stack(
            token_contour_list, dim=1
        )  # [n_agent, n_step, 4, 2]
        if len(token_all) > 0:
            token_all = torch.stack(token_all, dim=1)  # [n_agent, n_step, 6, 4, 2]

        expected_T = (n_step - 1) // self.shift

        # sanity check
        assert tuple(token_index.shape) == (
            n_agent,
            expected_T,
        ), f"Invalid token_index shape, got {token_index.shape}"
        assert tuple(token_contour.shape) == (
            n_agent,
            expected_T,
            token_contour_dim,
            feat_dim,
        ), f"Invalid token_contour shape, got {token_contour.shape}"

        return token_index, token_contour, token_all

    @staticmethod
    def _tokenize_map(data):

        data["map_polygon"]["type"] = data["map_polygon"]["type"].to(torch.uint8)
        data["map_point"]["type"] = data["map_point"]["type"].to(torch.uint8)
        pt2pl = data[("map_point", "to", "map_polygon")]["edge_index"]
        pt_type = data["map_point"]["type"].to(torch.uint8)
        pt_side = torch.zeros_like(pt_type)
        pt_pos = data["map_point"]["position"][:, :2]
        data["map_point"]["orientation"] = wrap_angle(data["map_point"]["orientation"])
        pt_heading = data["map_point"]["orientation"]
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polyline_side = []
        pl_idx_list = []
        split_polygon_type = []
        data["map_point"]["type"].unique()

        for i in sorted(np.unique(pt2pl[1])):  # number of polygons in the scenario
            index = pt2pl[
                0, pt2pl[1] == i
            ]  # index of points which belongs to i-th polygon
            polygon_type = data["map_polygon"]["type"][i]
            cur_side = pt_side[index]
            cur_type = pt_type[index]
            cur_pos = pt_pos[index]
            cur_heading = pt_heading[index]

            for side_val in np.unique(cur_side):
                for type_val in np.unique(cur_type):
                    if type_val == 13:
                        continue
                    indices = np.where((cur_side == side_val) & (cur_type == type_val))[
                        0
                    ]
                    if len(indices) <= 2:
                        continue
                    split_polyline = interplating_polyline(
                        cur_pos[indices].numpy(), cur_heading[indices].numpy()
                    )
                    if split_polyline is None:
                        continue
                    new_cur_type = cur_type[indices][0]
                    new_cur_side = cur_side[indices][0]
                    map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                    new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                    new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                    cur_pl_idx = torch.Tensor([i])
                    new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                    split_polyline_pos.append(split_polyline[..., :2])
                    split_polyline_theta.append(split_polyline[..., 2])
                    split_polyline_type.append(new_cur_type)
                    split_polyline_side.append(new_cur_side)
                    pl_idx_list.append(new_cur_pl_idx)
                    split_polygon_type.append(map_polygon_type)

        split_polyline_pos = torch.cat(split_polyline_pos, dim=0)
        split_polyline_theta = torch.cat(split_polyline_theta, dim=0)
        split_polyline_type = torch.cat(split_polyline_type, dim=0)
        split_polyline_side = torch.cat(split_polyline_side, dim=0)
        split_polygon_type = torch.cat(split_polygon_type, dim=0)
        pl_idx_list = torch.cat(pl_idx_list, dim=0)

        data["map_save"] = {}
        data["pt_token"] = {}
        data["map_save"]["traj_pos"] = split_polyline_pos
        data["map_save"]["traj_theta"] = split_polyline_theta[
            :, 0
        ]  # torch.arctan2(vec[:, 1], vec[:, 0])
        data["map_save"]["pl_idx_list"] = pl_idx_list
        data["pt_token"]["type"] = split_polyline_type
        data["pt_token"]["side"] = split_polyline_side
        data["pt_token"]["pl_type"] = split_polygon_type
        data["pt_token"]["num_nodes"] = split_polyline_pos.shape[0]

        return data
