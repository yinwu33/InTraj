from pathlib import Path

import numpy as np

from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneMarkType


class AV2MapVisualizer:
    def __init__(self):
        self.dataset_dir = "data"  # TODO: from config

    def _try_get_map_fpath(self, seq_id) -> Path:
        possible_splits = ["train", "val", "test"]
        for split in possible_splits:
            map_fpath = Path(
                self.dataset_dir
                + f"/{split}/{seq_id}"
                + f"/log_map_archive_{seq_id}.json"
            )
            if map_fpath.exists():
                return map_fpath
        raise FileNotFoundError(f"Map file for seq_id {seq_id} not found in any split.")

    def show_map(self, ax, seq_id: str, show_freespace=True):
        static_map_path = self._try_get_map_fpath(seq_id)
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(
                drivable_area.xyz[:, 0],
                drivable_area.xyz[:, 1],
                color="grey",
                alpha=0.2,
            )

        # ~ lane segments
        # print('num lane segs: ', len(static_map.vector_lane_segments),
        #       [x for x in static_map.vector_lane_segments.keys()])
        # print("Num lanes: ", len(static_map.vector_lane_segments))
        for lane_segment in static_map.vector_lane_segments.values():
            # print('left pts: ', lane_segment.left_lane_boundary.xyz.shape,
            #       'right pts: ', lane_segment.right_lane_boundary.xyz.shape)

            if lane_segment.lane_type == "VEHICLE":
                lane_clr = "blue"
            elif lane_segment.lane_type == "BIKE":
                lane_clr = "green"
            elif lane_segment.lane_type == "BUS":
                lane_clr = "orange"
            else:
                assert False, "Wrong lane type"

            # if lane_segment.is_intersection:
            #     lane_clr = 'yellow'

            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color=lane_clr, alpha=0.1)

            for boundary in [
                lane_segment.left_lane_boundary,
                lane_segment.right_lane_boundary,
            ]:
                ax.plot(
                    boundary.xyz[:, 0],
                    boundary.xyz[:, 1],
                    linewidth=1,
                    color="grey",
                    alpha=0.3,
                )

            # cl = static_map.get_lane_segment_centerline(lane_segment.id)
            # ax.plot(cl[:, 0], cl[:, 1], linestyle='--', color='magenta', alpha=0.1)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate(
                [pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)]
            )
            # plt.plot(edge[:, 0], edge[:, 1], color='orange', alpha=0.75)
            ax.fill(edge[:, 0], edge[:, 1], color="orange", alpha=0.2)
            # for edge in [ped_xing.edge1, ped_xing.edge2]:
            #     ax.plot(edge.xyz[:, 0], edge.xyz[:, 1], color='orange', alpha=0.5, linestyle='dotted')

    def show_map_clean(self, ax, seq_id: str, show_freespace=True):
        static_map_path = self._try_get_map_fpath(seq_id)
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(
                drivable_area.xyz[:, 0],
                drivable_area.xyz[:, 1],
                color="grey",
                alpha=0.2,
            )

        # ~ lane segments
        # print("Num lanes: ", len(static_map.vector_lane_segments))
        for lane_id, lane_segment in static_map.vector_lane_segments.items():
            lane_clr = "grey"
            polygon = lane_segment.polygon_boundary
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="whitesmoke",
                alpha=1.0,
                edgecolor=None,
                zorder=0,
            )

            # centerline
            centerline = static_map.get_lane_segment_centerline(lane_id)[
                :, 0:2
            ]  # use xy
            ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                alpha=0.1,
                color="grey",
                linestyle="dotted",
                zorder=1,
            )

            # lane boundary
            for boundary, mark_type in [
                (lane_segment.left_lane_boundary.xyz, lane_segment.left_mark_type),
                (lane_segment.right_lane_boundary.xyz, lane_segment.right_mark_type),
            ]:

                clr = None
                width = 1.0
                if mark_type in [
                    LaneMarkType.DASH_SOLID_WHITE,
                    LaneMarkType.DASHED_WHITE,
                    LaneMarkType.DOUBLE_DASH_WHITE,
                    LaneMarkType.DOUBLE_SOLID_WHITE,
                    LaneMarkType.SOLID_WHITE,
                    LaneMarkType.SOLID_DASH_WHITE,
                ]:
                    clr = "white"
                    zorder = 3
                    width = width
                elif mark_type in [
                    LaneMarkType.DASH_SOLID_YELLOW,
                    LaneMarkType.DASHED_YELLOW,
                    LaneMarkType.DOUBLE_DASH_YELLOW,
                    LaneMarkType.DOUBLE_SOLID_YELLOW,
                    LaneMarkType.SOLID_YELLOW,
                    LaneMarkType.SOLID_DASH_YELLOW,
                ]:
                    clr = "gold"
                    zorder = 4
                    width = width * 1.1

                style = "solid"
                if mark_type in [
                    LaneMarkType.DASHED_WHITE,
                    LaneMarkType.DASHED_YELLOW,
                    LaneMarkType.DOUBLE_DASH_YELLOW,
                    LaneMarkType.DOUBLE_DASH_WHITE,
                ]:
                    style = (0, (5, 10))  # loosely dashed
                elif mark_type in [
                    LaneMarkType.DASH_SOLID_YELLOW,
                    LaneMarkType.DASH_SOLID_WHITE,
                    LaneMarkType.DOUBLE_SOLID_YELLOW,
                    LaneMarkType.DOUBLE_SOLID_WHITE,
                    LaneMarkType.SOLID_YELLOW,
                    LaneMarkType.SOLID_WHITE,
                    LaneMarkType.SOLID_DASH_WHITE,
                    LaneMarkType.SOLID_DASH_YELLOW,
                ]:
                    style = "solid"

                if (clr is not None) and (style is not None):
                    ax.plot(
                        boundary[:, 0],
                        boundary[:, 1],
                        color=clr,
                        alpha=1.0,
                        linewidth=width,
                        linestyle=style,
                        zorder=zorder,
                    )

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate(
                [pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)]
            )
            ax.fill(edge[:, 0], edge[:, 1], color="yellow", alpha=0.1, edgecolor=None)
