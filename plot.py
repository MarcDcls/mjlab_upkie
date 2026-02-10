import json
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser(description="Plot logged data from sim.py")
parser.add_argument("--repository", type=str, default="logs/sim/", help="Path to the log repository")
args = parser.parse_args()


for filename in os.listdir(args.repository):
    if filename.endswith(".json"):
        with open(os.path.join(args.repository, filename), "r") as f:
            data = json.load(f)
        
        timestamps = data["timestamps"]
        left_hip_obs = data["left_hip"]["observation"]
        left_hip_act = data["left_hip"]["action"]
        left_knee_obs = data["left_knee"]["observation"]
        left_knee_act = data["left_knee"]["action"]
        right_hip_obs = data["right_hip"]["observation"]
        right_hip_act = data["right_hip"]["action"]
        right_knee_obs = data["right_knee"]["observation"]
        right_knee_act = data["right_knee"]["action"]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, left_hip_obs, label="Left Hip Observation")
        plt.plot(timestamps, left_hip_act, label="Left Hip Action")
        plt.title("Left Hip")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(timestamps, left_knee_obs, label="Left Knee Observation")
        plt.plot(timestamps, left_knee_act, label="Left Knee Action")
        plt.title("Left Knee")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()    

        plt.subplot(2, 2, 3)
        plt.plot(timestamps, right_hip_obs, label="Right Hip Observation")
        plt.plot(timestamps, right_hip_act, label="Right Hip Action")
        plt.title("Right Hip")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(timestamps, right_knee_obs, label="Right Knee Observation")
        plt.plot(timestamps, right_knee_act, label="Right Knee Action")
        plt.title("Right Knee")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.show()

        left_wheel_obs = data["left_wheel"]["observation"]
        left_wheel_act = data["left_wheel"]["action"]
        right_wheel_obs = data["right_wheel"]["observation"]
        right_wheel_act = data["right_wheel"]["action"]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(timestamps, left_wheel_obs, label="Left Wheel Observation")
        plt.plot(timestamps, left_wheel_act, label="Left Wheel Action")
        plt.title("Left Wheel")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(timestamps, right_wheel_obs, label="Right Wheel Observation")
        plt.plot(timestamps, right_wheel_act, label="Right Wheel Action")
        plt.title("Right Wheel")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.show()