from roboflow import Roboflow

rf = Roboflow(api_key="xXTCES4dsTphZGmPM4zO")
project = rf.workspace("ibm-space-rover").project("planets-extlk")
version = project.version(1)
dataset = version.download("yolov11")
