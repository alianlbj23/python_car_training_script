from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import sys
from entity.State import State
import json

class TrainingAgent(Node):
    def __init__(self) -> None:
        super().__init__("TrainingAgent")
        self.get_logger().info("TrainingAgent start")
        self.subscriber_from_unity = self.create_subscription(String, "/Unity2TrainingAgent", self.receive_state, 10)
        self.publisher_to_unity = self.create_publisher(Float32MultiArray, '/TrainingAgent2Unity', 10)

        self.subscriber_from_trainer = self.create_subscription(Float32MultiArray, "/Trainer2TrainingAgent", self.receive_action, 10)
        self.publisher_to_trainer = self.create_publisher(String, '/TrainingAgent2Trainer', 10)
        
        self.state = State()
        self.unityState = ""
    


    def receive_state(self, msg):
        data = self.__parse_json(msg.data)
        if not data["isFirst"]:
            self.publisher_to_trainer.publish(msg)
        else:
            self.publisher_to_unity.publish()
    
    def receive_action(self, msg):
        pass


if __name__ == '__main__':
    rclpy.init()
    
    mode = 'train'
    training_agent = TrainingAgent()
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(training_agent)
    exe.spin()

    rclpy.shutdown()
    sys.exit(0)