from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.map.lane_segment import LaneType, LaneMarkType

_AGENT_TYPE_MAP = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.MOTORCYCLIST: 2,
    ObjectType.CYCLIST: 3,
    ObjectType.BUS: 4,
    
    # static objects are labeled as 6
    
    ObjectType.UNKNOWN: 5,
}

_LANE_TYPE_MAP = {
    LaneType.VEHICLE: 0,
    LaneType.BIKE: 1,
    LaneType.BUS: 2,
}


_AGENT_SCORE_TYPE_MAP = {
    "frag": 0,
    TrackCategory.TRACK_FRAGMENT: 0,
    "unscore": 1,
    TrackCategory.UNSCORED_TRACK: 1,
    "score": 2,
    TrackCategory.SCORED_TRACK: 2,
    "focal": 3,
    TrackCategory.FOCAL_TRACK: 3,
    "ego": 4,
}

_LANE_MARK_TYPE = {
    LaneMarkType.DASH_SOLID_YELLOW: 0,
    LaneMarkType.DASH_SOLID_WHITE: 0,
    LaneMarkType.DASHED_WHITE: 0,
    LaneMarkType.DASHED_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_YELLOW: 0,
    LaneMarkType.DOUBLE_DASH_WHITE: 0,
    LaneMarkType.DOUBLE_SOLID_YELLOW: 1,
    LaneMarkType.DOUBLE_SOLID_WHITE: 1,
    LaneMarkType.SOLID_YELLOW: 1,
    LaneMarkType.SOLID_WHITE: 1,
    LaneMarkType.SOLID_DASH_WHITE: 1,
    LaneMarkType.SOLID_DASH_YELLOW: 1,
    LaneMarkType.SOLID_BLUE: 1,
}
