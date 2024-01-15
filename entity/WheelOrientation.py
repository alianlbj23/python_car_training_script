from pydantic import BaseModel

# around ROS2 z axis, left +, right -, up 0, down 180
class WheelOrientation(BaseModel):
    left_front: float = 0
    right_front: float = 0