import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class AINode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")  # ros2Ai #unity2Ros
        self.subscriber = self.create_subscription(String, "/unity2Ros", self.receive_data_from_ros, 10)

        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, '/ros2Unity', 10)  # Ai2ros #ros2Unity
        self.unityState = ""

    def publish2Ros(self, data):
        self.data2Ros = Float32MultiArray()
        self.data2Ros.data = data
        self.publisher_Ai2ros.publish(self.data2Ros)

    def receive_data_from_ros(self, msg):
        print("received")
        self.unityState = msg.data
        
    def return_unity_state(self):
        while len(self.unityState) == 0:
            pass
        return self.unityState
   
    def not_connect_to_Unity(self):
        if len(self.unityState) == 0:
            return True
        else:
            return False
    
    def connect_Unity(self, state, action_Unity_Unity_adaptor):
        self.unityState = self.return_unity_state()
        state.update(self.unityState, action_Unity_Unity_adaptor)
        return state.current_car_state_training