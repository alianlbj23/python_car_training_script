import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from AINode import AINode


class ROS2NodeManager:
    def __init__(self):
        self.nodes = []
        rclpy.init()

    def add_node(self, node):
        self.nodes.append(node)
    
    def start(self):
        ROSnode_thread = threading.Thread(target=self.spin_pros)
        ROSnode_thread.start()

    def spin_pros(self):
        exe = rclpy.executors.SingleThreadedExecutor()
        for node in self.nodes:
            exe.add_node(node)
            exe.spin()
        rclpy.shutdown()
        sys.exit(0)