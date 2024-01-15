from pydantic import BaseModel

class ControlSignal(BaseModel):
    wheel_vel: float  # rad/s
    steering_angle: float  # degree, left: -, right: +