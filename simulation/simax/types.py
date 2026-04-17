from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class Vector(BaseModel):
    x: float
    y: float
    z: float


class TrajectoryPoint(BaseModel):
    time: int  # ms
    position: Vector
    heading: float


class Trajectory(BaseModel):
    points: List[TrajectoryPoint] = []


class TrajectoryWithId(BaseModel):
    trajectory: Trajectory
    object_id: int


class ObjectCategory(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"


class ObjectState(str, Enum):
    MOVING = "moving"
    STOPPED = "stopped"
    PARKED = "parked"
    UNKNOWN = "unknown"


class VehicleState(BaseModel):
    object_id: int
    position: Vector
    heading: float
    velocity: float
    object_category: ObjectCategory
    object_state: ObjectState
    length: float
    width: float


class VehicleStates(BaseModel):
    time: int
    states: List[VehicleState]


class MapInitRequest(BaseModel):
    map_file_path: str
