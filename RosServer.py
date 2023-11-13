#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys

from std_msgs.msg import Float32MultiArray
import threading


class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")
        self.subsvriber_ = self.create_subscription(Float32MultiArray, "ros2Ai", self.receive_data_from_ros, 10)
        self.unityState = None

    def returnData(self):
        return self.unityState

    
    
    def receive_data_from_ros(self, msg):
        self.unityState = msg.data
        print(self.unityState)
        
        

def spin_pros(node):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(node)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)
    


def main():
    rclpy.init()
    node = AiNode()
    pros = threading.Thread(target=spin_pros, args=(node,))
    pros.start()    
    

    

    
    
# if __name__ == "__main__":
#     main()