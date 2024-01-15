from pydantic import BaseModel
# around car wheel axis, front: +, back: -, r/s
class WheelAngularVel(BaseModel):
    left_back: float = 0
    left_front: float = 0
    right_back: float = 0
    right_front: float = 0