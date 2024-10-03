import base64
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import subprocess
import queue
import threading
import pyttsx3

app = Flask(__name__)
detection_queue = queue.Queue()
stop_threads = False  # Flag to stop threads when the application exits

def detect_objects():
    global stop_threads

    thres = 0.45  # Threshold to detect object
    known_width = 20  # Width of the object in the real world (in centimeters)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 320)
    cap.set(10, 70)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    engine = pyttsx3.init()

    while not stop_threads:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0)

        curr_objects = set()

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                curr_objects.add(classNames[classId - 1])
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                
                # Calculate distance of object (simple approximation)
                width_pixel = box[2]  # Width of the bounding box in pixels
                distance = (known_width * 640) / (2 * width_pixel * np.tan(62 * np.pi / (2 * 180)))
                detection_queue.put((classNames[classId - 1].capitalize(), distance))  # Add object name and distance to queue

                # Display object name and distance outside the rectangle
                cv2.putText(img, f"{classNames[classId - 1].capitalize()}: {distance:.2f} cm", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        # Check for objects that disappeared in the current frame and remove them from the queue
        if detection_queue.qsize() > 0:
            for _ in range(detection_queue.qsize()):
                detection_queue.get()

        cv2.imshow("Output", img)
        cv2.waitKey(1)

    cap.release()  # Release the camera when the thread exits

def voice_output():
    global stop_threads
    engine = pyttsx3.init()
    while not stop_threads:
        if not detection_queue.empty():
            result = detection_queue.get()
            engine.say(f"{result[0]} is approximately {result[1]:.2f} centimeters away")
            engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_objects', methods=['POST'])
def start_detection():
    global stop_threads
    if detection_queue.empty():
        stop_threads = False  # Reset the stop_threads flag
        threading.Thread(target=detect_objects).start()
        threading.Thread(target=voice_output).start()  # Start voice output thread
        return 'Object detection process started.'
    else:
        return 'Object detection process is already running.'

@app.route('/get_detection_result', methods=['GET'])
def get_detection_result():
    if not detection_queue.empty():
        result = detection_queue.get()
        return f"{result[0]} is approximately {result[1]:.2f} centimeters away"
    else:
        return 'No object detection results available.'

if __name__ == '__main__':
    app.run(debug=True)
