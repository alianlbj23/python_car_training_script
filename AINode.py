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

        self.publisher_Ai2ros = self.create_publisher(String, '/trainingData', 10)  # Ai2ros #ros2Unity
        self.unityState = ""

    def receive_data_from_ros(self, msg):
        print("received")
        self.publisher_Ai2ros.publish(msg)