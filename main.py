import asyncio
import base64
import json
import time
from sympy import true
import websockets
import socket
import queue
import threading
import os
import re
import sys
import subprocess
import serial
import struct

import cv2
from ultralytics import YOLO
import numpy as np

from rgb_camera import CAMERA_DRIVER
from web_app_files.web_app_launcher import run_web_app, get_server_status

### CONFIG ###
global_settings = {
    "T" : "SETTINGS",
    "Obstacle Stop Distance (cm)" : 20,
    "Rover l Max Speed" : 1000,
    "Rover l Min Speed" : -1000,
    "Rover l Base Speed" : 500,
    "Line Detect Thres" : 500,
    "Line PID Kp" : 0.1,
    "Line PID Kd" : 0.05,
    "Checkpoint Adjustment (s)" : 0.5,
    "Scanning Speed (mm/s)" : 5,
    "Spraying Speed (mm/s)" : 10,
    "YOLO Model path" : "/home/rasp/project/pest_robot_program/image_models/pest_model.pt",
    # "YOLO Model path" : "image_models/yolov8n.pt",
    "YOLO Model Conf" : 0.5,
    "YOLO model Healthy Value" : 0,
    "YOLO model Pest1 Value" : 1,
    "YOLO model Pest2 Value" : 2,
}

class SERVER_DATA_STRUCT:
    ws_client_connected = False
    ws_camera_stream_on = False
    ws_client_handler:websockets.ServerConnection
    ws_camera_stream_task:asyncio.Task
    wss_camera_feed_q = queue.Queue(maxsize=10)
    wss_data_tx_q = queue.Queue(maxsize=10)

class GLOBAL_DATA_STRUCT:
    is_task_running = False

### useful functions
class JsonConfigManager:
    def __init__(self, config_dict, file_path):
        """
        Initializes the configuration manager.
        
        Args:
            config_dict (dict): The global dictionary (defaults). Modified IN-PLACE.
            file_path (str): The path to the JSON file.
        """
        self.data = config_dict  # Direct reference (no deepcopy)
        self.file_path = file_path

    def save(self):
        """Saves the linked dictionary to the JSON file."""
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, indent=4)

    def load_and_sync(self):
        """
        Loads the JSON file and modifies the linked dictionary in-place.
        - Overwrites defaults with saved values.
        - Keeps default values for missing keys.
        - Ignores outdated/extra keys from the file.
        """
        if not os.path.exists(self.file_path):
            print(f"[{self.file_path}] Not found. Creating new file with default values.")
            self.save()
            return

        with open(self.file_path, 'r', encoding='utf-8') as file:
            try:
                loaded_dict = json.load(file)
            except json.JSONDecodeError:
                loaded_dict = {}

        loaded_keys = set(loaded_dict.keys())
        current_keys = set(self.data.keys())
        needs_saving = loaded_keys != current_keys

        # Modify the global dictionary in-place
        for key in current_keys:
            if key in loaded_dict:
                # File has the key: overwrite the default with the saved value
                self.data[key] = loaded_dict[key]
            else:
                # File is missing the key: the global dict already has the default value
                print(f"Added new missing key to {self.file_path}: '{key}'")

        removed_keys = loaded_keys - current_keys
        if removed_keys:
            print(f"Ignored outdated keys from {self.file_path}: {', '.join(removed_keys)}")

        # Clean up the file if it had extra/missing keys
        if needs_saving:
            self.save()
            print(f"[{self.file_path}] Keys synced and file updated.")

def check_for_updates_and_restart():
    print("Checking GitHub for updates...")
    try:
        # 1. Fetch the latest info from the remote repository
        subprocess.run(["git", "fetch", "--all"], check=True, capture_output=True)
        
        # 2. Check if the local branch is behind the remote branch
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True)
        
        if "Your branch is behind" in status.stdout:
            print("Update found! Downloading latest code...")
            
            # 3. Force update to exactly match the remote (prevents merge conflicts)
            # CHANGE 'main' TO 'master' if your repo uses the older naming!
            subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
            
            print("Update complete. Restarting program to apply changes...")
            
            # 4. Instantly restart the Python script so the new code runs
            try:
                os.closerange(3, 65535) # Close ports just like your restart function
            except:
                pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
        else:
            print("App is fully up to date. Starting normally...")
            
    except Exception as e:
        print(f"Failed to check for updates: {e}")
        print("Continuing with the current version...")


class SerialComHandler(threading.Thread):
    def __init__(self, port='/dev/serial0', baudrate=115200):
        super().__init__(daemon=True)
        self._stop_worker = threading.Event()
        self._serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=1)

        self._tx_queue = queue.Queue()
        self.rx_queue = queue.Queue()

    def stop(self):
        self._stop_worker.set()

    def tx_data(self, data:bytes):
        self._tx_queue.put_nowait(data)

    def get_rx_queue(self):
        return self.rx_queue
    
    def run(self):
        print("Serial Handler Started!")

        while not self._stop_worker.is_set():
            try:
                pkt = self._tx_queue.get_nowait()
                self._serial_port.write(pkt)
                self._tx_queue.task_done()

            except queue.Empty:
                pass

            if self._serial_port.in_waiting > 0:
                rx_bytes = self._serial_port.readline()
                rx_str = rx_bytes.decode('utf-8', 'ignore').rstrip()
                self.rx_queue.put_nowait(rx_str)

            time.sleep(0.01)

#####################

### image hanlder thread
class ImageProcessWorker(threading.Thread):
    is_scanning = False
    detected_pests = 0

    def __init__(self, server_st: SERVER_DATA_STRUCT):
        super().__init__(daemon=True)
        self._stop_worker = threading.Event()
        self._ws_server = server_st

        self.yolo_model = YOLO(global_settings["YOLO Model path"], 'detect')
        self.yolo_model_class_ids = [global_settings["YOLO model Healthy Value"], global_settings["YOLO model Pest1 Value"], global_settings["YOLO model Pest2 Value"]]
        self.pest_texts = {
            0: self.yolo_model.names[self.yolo_model_class_ids[0]],
            1: self.yolo_model.names[self.yolo_model_class_ids[1]],
            2: self.yolo_model.names[self.yolo_model_class_ids[2]],
            3: f"{self.yolo_model.names[self.yolo_model_class_ids[1]]} {self.yolo_model.names[self.yolo_model_class_ids[2]]}" # Added a space for readability
        }
        
    def stop(self):
        self._stop_worker.set()

    def run(self):
        print("Image Processor worker Started!")

        while not self._stop_worker.is_set():
            _, cam_frame = camera.capture()

            if self.is_scanning is True:
                yolo_results = self.yolo_model.predict(cam_frame, imgsz=480, conf=global_settings["YOLO Model Conf"])
                cam_frame = yolo_results[0].plot() # type: ignore

                # 1. Vectorize class extraction (eliminates the slow Python 'for' loop)
                if yolo_results[0].boxes: # type: ignore
                    # Convert YOLO tensor to a Python set for O(1) lookups
                    detected_classes = set(yolo_results[0].boxes.cls.int().tolist()) # type: ignore
                else:
                    detected_classes = set()

                # 2. Update state using bitwise OR 
                # (State mask: 1 = Pest A, 2 = Pest B, 3 = Both)
                if self.yolo_model_class_ids[1] in detected_classes:
                    self.detected_pests |= 1

                if self.yolo_model_class_ids[2] in detected_classes:
                    self.detected_pests |= 2

            ## send to the web app
            if self._ws_server.ws_camera_stream_on:
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                    _, img_buf = cv2.imencode('.jpg', cam_frame, encode_param)
                    img_buf = base64.b64encode(img_buf)
                    img_buf = 'cam'.encode() + img_buf
                    self._ws_server.wss_camera_feed_q.put_nowait(img_buf)
                
                except queue.Full:
                    pass

            time.sleep(0.005)

    def start_scan(self):
        ws_tx_str(json.dumps({"T":"TER", "D": "Plant Scanning Started"}))

        self.detected_pests = 0
        self.is_scanning = True

    def stop_scan(self):
        global global_data
        self.is_scanning = False 

        ## update global values
        scan_detected_pest_text = self.pest_texts.get(self.detected_pests, self.pest_texts[0])
        scan_detected_pest_type = self.detected_pests

        ws_tx_str(json.dumps({"T":"TER", "D": "Plant Scanning Completed"}))

        return (scan_detected_pest_type, scan_detected_pest_text)

class RoverControllerWorker(threading.Thread):
    is_task_running = False
    is_manual_control_act = False
    mannual_cmds = None
    quick_actions = None
    mannual_line_follow_flag = False
    mannual_pump1 = False
    mannual_pump2 = False


    is_check_point = False
    curr_state = 0
    ack_recv = False

    nxt_batt_acq = 0

    def __init__(self, serial_port:SerialComHandler, img_worker:ImageProcessWorker):
        super().__init__(daemon=True)
        self._stop_worker = threading.Event()

        self.serial_port = serial_port
        self.img_worker = img_worker
        self.task_manager = RoverPestTask(self)

    def stop(self):
        self._stop_worker.set()
    
    def run(self):
        global global_data
        print("Rover worker Started!")

        ws_tx_str(json.dumps({"T":"TER", "D": "STM32 Connection Checking"}))
        self._send_cmd_test_connection()
        self._get_wait_ack()
        ws_tx_str(json.dumps({"T":"TER", "D": "STM32 Connected"}))
        rover_controller._send_cmd_buzzer(500)

        while not self._stop_worker.is_set():
            self._handle_rx_msg()

            ## accuruire batt
            # if time.time() > self.nxt_batt_acq:
            #     self.nxt_batt_acq = time.time() + 5 ## 5sec
            #     self._send_cmd_batt()

            if self.quick_actions is not None:
                if self.is_task_running:
                    self.start_task() ## stop task if running

                cmd = self.quick_actions["CMD"]
                self.quick_actions = None

                if cmd == "CALIBRATE_IR":
                    self._send_cmd_ir_calib()
                    self._get_wait_ack()

                elif cmd == "RESET_PLATFORM":
                    self._send_cmd_zero_plat()
                    self._get_wait_ack()

                elif cmd == "TOGGLE_PUMP_1":
                    self.mannual_pump1 = not self.mannual_pump1
                    self._send_cmd_pump(1, self.mannual_pump1)
                    self._get_wait_ack()
                    
                elif cmd == "TOGGLE_PUMP_2":
                    self.mannual_pump2 = not self.mannual_pump2
                    self._send_cmd_pump(2, self.mannual_pump2)
                    self._get_wait_ack()

                elif cmd == "TEST_LINE_FOLLOW":
                    self.mannual_line_follow_flag = not self.mannual_line_follow_flag
                    if self.mannual_line_follow_flag:
                        self._send_cmd_follow_line()
                    else:
                        self._send_cmd_stop()

            ### manual control lock
            if self.is_manual_control_act:
                if self.is_task_running:
                    self.start_task() ## stop task if running

                if self.mannual_cmds is not None:
                    if self.mannual_cmds["ACT"] == "PRESS":
                        motor_spd:int = int(self.mannual_cmds["SPD"]) * 10
                        rover_dir = self.mannual_cmds["DIR"]
                        motor_out = (motor_spd, motor_spd) if rover_dir == "UP" else \
                                    (-motor_spd, -motor_spd) if rover_dir == "DOWN" else \
                                    (motor_spd, -motor_spd) if rover_dir == "LEFT" else \
                                    (-motor_spd, motor_spd) if rover_dir == "RIGHT" else \
                                    (0, 0)
                        self._send_cmd_motor(*motor_out)
                    else:
                        self._send_cmd_motor(0,0)
                    
                    self.mannual_cmds = None
                else:
                    time.sleep(0.05)
            
            ### task run
            if self.is_task_running:
                self.task_manager.run_task()
            
            time.sleep(0.02)


    def set_action(self, type:dict):
        cmd = type["CMD"]

        if cmd == "MCTRL_LOCK":
            self.is_manual_control_act = type["STATE"]
        else:
            self.quick_actions = type

    def manual_control(self, data):
        self.mannual_cmds = data

    def start_task(self):
        if not global_data.is_task_running:
            ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Task Running..."}))
            ws_tx_str(json.dumps({"T":"TER", "D": "Task Started"}))

            self.task_manager.reset_iteration()
            global_data.is_task_running = True
            self.is_task_running = True

        else:
            global_data.is_task_running = False
            self.is_task_running = False

            self.img_worker.stop_scan()
            ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Task Stopping..."}))
            ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Task Completed!"}))
            ws_tx_str(json.dumps({"T":"TER", "D": "Task Completed!"}))
        
    def stop_task(self):
        self.is_task_running = False

    def _handle_rx_msg(self): ## json process shutdown start commands
        try:
            rx_msg = self.serial_port.rx_queue.get_nowait()
            rx_json = json.loads(rx_msg)
            cmd_str = rx_json["cmd"]

            if cmd_str == "OK":
                self.ack_recv = True

            elif cmd_str == "CHKP": # {"cmd": "CHKP"}
                self.is_check_point = True

            elif cmd_str == "BLOCK": # {"cmd": "BLOCK"}
                ws_tx_str(json.dumps({"T":"ALERT", "D": "OBJ_DETECTED"}))

            elif cmd_str == "CLEAR": # {"cmd": "CLEAR"}
                ws_tx_str(json.dumps({"T":"ALERT", "D": "OBJ_CLEARED"}))

            elif cmd_str == "batt": # {"cmd": "batt", "mvol": %ld}
                batt_voltage = int(rx_json["mvol"])
                batt_precentage = (batt_voltage - 10500) * 100 / (12400 - 10500)
                ws_tx_str(json.dumps({"T":"BATT", "V": int(batt_precentage)}))

            elif cmd_str == "button": # {"cmd": "button", "state": %d} BTN_NONE = 0, BTN_SINGLE_PRESS, BTN_LONG_PRESS
                btn_state = int(rx_json["state"])

                if btn_state == 1:
                    self.start_task()
                elif btn_state == 2:
                    ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Shutting Down..."}))
                    ws_tx_str(json.dumps({"T":"TER", "D": "SHUTDOWN ROVER"}))
                    self._send_cmd_stm_reset()
                    shutdown_raspberry_pi()
                
            self.serial_port.rx_queue.task_done()
            
        except:
            pass
        
    def _get_serial_ack(self):
        temp = self.ack_recv
        self.ack_recv = False
        return temp
    
    def _get_wait_ack(self, timeout_ms = 1000):
        timeout_in = (time.time() * 1000) + timeout_ms
        while not self.ack_recv and timeout_in > time.time() * 1000:
            self._handle_rx_msg()
            time.sleep(0.02)

        temp = self.ack_recv
        self.ack_recv = False
        return temp
        
    def _send_json_pkt(self, data_dict):
        json_string = json.dumps(data_dict)
        json_payload = json_string + "\n"
        byte_pkt = json_payload.encode('utf-8')

        self._get_serial_ack()
        self.serial_port.tx_data(byte_pkt)
    
    def _send_cmd_stepper(self, position:int, rate:int):
        rate = 50 if rate > 50 else 1 if rate <= 0 else rate
        self._send_json_pkt({"cmd":"stepper","P":position,"R":rate})

    def _send_cmd_motor(self, Lspeed:int, Rspeed:int):
        self._send_json_pkt({"cmd":"motor","L":Lspeed,"R":Rspeed})

    def _send_cmd_servo(self, angle:int):
        self._send_json_pkt({"cmd":"servo","V":angle})
        
    def _send_cmd_pump(self, pump:int, state:int):
        self._send_json_pkt({"cmd":"pump","P":pump,"S":state})

    def _send_cmd_buzzer(self, tms:int):
        self._send_json_pkt({"cmd":"buzz","V":tms})

    def _send_cmd_test_connection(self):
        self._send_json_pkt({"cmd":"OK"})

    def _send_cmd_batt(self):
        self._send_json_pkt({"cmd":"batt"})

    def _send_cmd_follow_line(self):
        self._send_json_pkt({"cmd":"linef"})
    
    def _send_cmd_ir_calib(self):
        self._send_json_pkt({"cmd":"calib"})

    def _send_cmd_zero_plat(self):
        self._send_json_pkt({"cmd":"plzero"})

    def _send_cmd_stm_reset(self):
        self._send_json_pkt({"cmd":"streset"})
    
    def _send_cmd_stop(self):
        self._send_json_pkt({"cmd":"stop"})

    def _send_cmd_settings(self):
        SETTINGS_STRUCT = '<h h h h h f f' ## 

        ws_tx_str(json.dumps({"T":"TER", "D": "Saving Settings"}))

        settings_bytes = struct.pack(SETTINGS_STRUCT, 
                                     int(global_settings["Obstacle Stop Distance (cm)"]), 
                                     int(global_settings["Rover l Max Speed"]), 
                                     int(global_settings["Rover l Min Speed"]), 
                                     int(global_settings["Rover l Base Speed"]), 
                                     int(global_settings["Line Detect Thres"]), 
                                     float(global_settings["Line PID Kp"]), 
                                     float(global_settings["Line PID Kd"]))
        
        settings_bytes = settings_bytes.ljust(32, b'\x00')

        json_string = json.dumps({"cmd":"sett","D": ""})
        byte_pkt = json_string.encode('utf-8')
        byte_pkt = byte_pkt[:-3] + settings_bytes + "}\n".encode('utf-8')

        self._get_serial_ack()
        self.serial_port.tx_data(byte_pkt)
        self._get_wait_ack()

        ws_tx_str(json.dumps({"T":"TER", "D": "Settings Updated"}))


class RoverPestTask():
    ## State Machine States
    START_UP = 0
    FOLLOW_PATH = 1
    ROTATE_PLATFORM = 2
    CAMERA_SCAN = 3
    SPRAY_CHEMICAL = 4
    CHK_PNT_FINISH = 5

    current_plant_pest = 0 ## 0 healthy, 1 pest(1), 2 pest(2), 3 both pest
    current_plant_count = 0

    ## servo vars
    SERVO_ANGLES = [90, 180, 0]## 90 front, 0 right, 180 left
    current_cam_angle = 0 ## 90 front, 0 right, 180 left

    ## stepper vars
    STEPPER_POINTS = [0, 210]
    current_step_point = 0

    def __init__(self, rover_worker:RoverControllerWorker):
        self.state = self.START_UP
        self.is_new_state = True
        self.rover_worker = rover_worker

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.is_new_state = True

    def reset_iteration(self):
        self.state = self.START_UP
        self.is_new_state = True
        self.current_plant_count = 0
        self.current_plant_pest = 0

    def run_task(self):
        if self.state == self.START_UP:
            self.rover_worker.is_check_point = False
            self.current_step_point = 0
            self.current_cam_angle = 0
            self.rover_worker._send_cmd_servo(self.SERVO_ANGLES[self.current_cam_angle])
            self.rover_worker._get_wait_ack()
            self.rover_worker._send_cmd_zero_plat()
            self.rover_worker._get_wait_ack()
            self.change_state(self.FOLLOW_PATH)

        elif self.state == self.FOLLOW_PATH: ## follow line till check point met
            if self.is_new_state:
                self.is_new_state = False

                self.rover_worker._send_cmd_follow_line()
                self.rover_worker._get_wait_ack()

            if self.rover_worker.is_check_point: ## check point detected and move to scanning
                self.rover_worker.is_check_point = False
                ws_tx_str(json.dumps({"T":"TER", "D": "Checkpoint Detected"}))
                
                self.rover_worker._send_cmd_stop()

                self.rover_worker._send_cmd_motor(global_settings["Rover l Base Speed"], global_settings["Rover l Base Speed"]) ## forward a bit to align with plant
                self.rover_worker._get_wait_ack()
                time.sleep(global_settings["Checkpoint Adjustment (s)"])
                self.rover_worker._send_cmd_motor(0,0)
                self.rover_worker._get_wait_ack()

                self.change_state(self.ROTATE_PLATFORM)

        elif self.state == self.ROTATE_PLATFORM:
            if self.is_new_state:
                self.is_new_state = False

                self.current_cam_angle += 1

                if self.current_cam_angle == 3: ## successfuly done plants on both sides
                    self.current_cam_angle = 0
                    self.change_state(self.CHK_PNT_FINISH) ## current check point finished

                else: ## turn to plant sides
                    self.rover_worker._send_cmd_servo(self.SERVO_ANGLES[self.current_cam_angle])
                    self.rover_worker._get_wait_ack()
                    self.change_state(self.CAMERA_SCAN)

        elif self.state == self.CAMERA_SCAN:
            if self.is_new_state:
                self.is_new_state = False

                if self.current_step_point == 1:
                    self.current_step_point = 0
                    self.rover_worker._send_cmd_stepper(self.STEPPER_POINTS[self.current_step_point], 40) ## reset platform to bottom
                    self.rover_worker._get_wait_ack()
                    
                self.rover_worker.img_worker.start_scan() ## start scanning image
                self.current_step_point = 1
                self.rover_worker._send_cmd_stepper(self.STEPPER_POINTS[self.current_step_point], global_settings["Scanning Speed (mm/s)"]) ## gradually move up platform
                
            if self.rover_worker._get_serial_ack(): ## wait untill platform completely moved
                time.sleep(0.5) ## settle time

                pest_type, pest_text = self.rover_worker.img_worker.stop_scan() ## stop and accquire results
                self.current_plant_pest = pest_type
                ws_tx_str(json.dumps({"T":"TER", "D": f"Plant {self.current_plant_count} | {pest_text}"}))

                self.change_state(self.SPRAY_CHEMICAL)
        
        elif self.state == self.SPRAY_CHEMICAL:
            if self.is_new_state:
                self.is_new_state = False

                if self.current_plant_pest == 0:
                    self.change_state(self.ROTATE_PLATFORM)
                    return

                else:
                    self.rover_worker._send_cmd_pump(1, (self.current_plant_pest in (1, 3)))
                    self.rover_worker._get_wait_ack()
                    self.rover_worker._send_cmd_pump(2, (self.current_plant_pest in (2, 3)))
                    self.rover_worker._get_wait_ack()

                    self.current_step_point = 0
                    self.rover_worker._send_cmd_stepper(self.STEPPER_POINTS[self.current_step_point], global_settings["Spraying Speed (mm/s)"]) ## gradually move platform to bottom
               
            if self.rover_worker._get_serial_ack(): ## wait untill platform completely moved
                time.sleep(0.2) ## settle time

                self.rover_worker._send_cmd_pump(1, 0)
                self.rover_worker._get_wait_ack()
                self.rover_worker._send_cmd_pump(2, 0)
                self.rover_worker._get_wait_ack()

                self.change_state(self.ROTATE_PLATFORM)

        elif self.state == self.CHK_PNT_FINISH:
            if self.is_new_state:
                self.is_new_state = False

                self.rover_worker._send_cmd_servo(self.SERVO_ANGLES[self.current_cam_angle])
                self.rover_worker._get_wait_ack()
                self.change_state(self.FOLLOW_PATH)

                    

def ws_tx_str(msg):
    try:
        server_data.wss_data_tx_q.put_nowait(msg)
    except queue.Full:
        pass

def ws_rx_data_handler(data):
    global server_data
    cmd_type = data["T"]

    if cmd_type == "GET_SETTINGS":
        ws_tx_str(json.dumps(global_settings))

    elif cmd_type == "LIVE":
        if not server_data.ws_camera_stream_on:
            ws_tx_str(json.dumps({"T":"TER", "D": "Camera Live ON"}))
            cleanup_queue(server_data.wss_camera_feed_q)
            server_data.ws_camera_stream_on = True
        else:
            ws_tx_str(json.dumps({"T":"TER", "D": "Camera Live OFF"}))
            server_data.ws_camera_stream_on = False

    elif cmd_type == "START":
        rover_controller.start_task()

    elif cmd_type == "MCTRL_LOCK":
        rover_controller.set_action({"CMD":"MCTRL_LOCK", "STATE": data["ACTIVE"]})
    
    elif cmd_type == "ACTION":
        rover_controller.set_action({"CMD": data["CMD"]})
    
    elif cmd_type == "MCTRL":
        rover_controller.manual_control(data)

    elif cmd_type == "STMRESET":
        rover_controller._send_cmd_stm_reset()

    elif cmd_type == "RESTART":
        ws_tx_str(json.dumps({"T":"TER", "D": "PROGRAM RESTARTING.."}))
        restart_python_program()
        
    elif cmd_type == "POWER":
        ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Shutting Down..."}))
        ws_tx_str(json.dumps({"T":"TER", "D": "SHUTDOWN ROVER"}))
        rover_controller._send_cmd_stm_reset()
        shutdown_raspberry_pi()

    elif cmd_type == "SETTINGS":
        global_settings.update(data)
        json_cfg.save()
        rover_controller._send_cmd_settings()


def cleanup_queue(x_q:queue.Queue):
    while not x_q.empty():
        try:
            x_q.get_nowait()
            x_q.task_done()  # Include this if you are using q.join() elsewhere
        except queue.Empty:
            break

async def ws_tx_hanlder_task(ws_t:SERVER_DATA_STRUCT):
    print("WS Tx Hanlder Started!")
    
    try:
        while True:
            if server_data.ws_client_connected and server_data.ws_client_handler is not None:
                try:
                    pkt = ws_t.wss_data_tx_q.get_nowait()
                    await server_data.ws_client_handler.send(pkt)
                    ws_t.wss_data_tx_q.task_done()
                    
                except queue.Empty:
                    pass

                if ws_t.ws_camera_stream_on is True:
                    try:
                        image_frame = ws_t.wss_camera_feed_q.get_nowait()
                        await server_data.ws_client_handler.send(image_frame)
                        ws_t.wss_camera_feed_q.task_done()
                        
                    except queue.Empty:
                        pass
            
            await asyncio.sleep(0.01)
    
    except asyncio.CancelledError:
        print("Cleaning WS Tx Task")
        cleanup_queue(server_data.wss_camera_feed_q)
        cleanup_queue(server_data.wss_data_tx_q)
        server_data.ws_camera_stream_on = False
        raise

    except Exception as e:
        print(f"WS task closed due to : {e}")

async def wshandler(websocket):
    global server_data
    server_data.ws_client_connected = True
    server_data.ws_client_handler = websocket
    
    try:
        async for message in websocket:
            try:
                ws_rx_data_handler(json.loads(message))
            except Exception as e:
                print(f"ERROR: WS rx data | {e}")
                
    except websockets.exceptions.ConnectionClosedOK:
        print("WS Client disconnected!")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WS Client connection closed with error: {e}")

    finally:
        server_data.ws_client_connected = False
        server_data.ws_camera_stream_on = False
        server_data.ws_client_handler = None # type: ignore
        print("WS Connection handler finished.")

async def start_ws_server():
    async with websockets.serve(wshandler, "0.0.0.0", 8000):
        print(f"App server started @ {socket.gethostname()}.local")
        loop = asyncio.get_running_loop()
        server_data.ws_camera_stream_task = loop.create_task(ws_tx_hanlder_task(server_data))
        await asyncio.Future()

def cleanup_resources():
    """Stops threads and cancels async tasks before exiting."""
    print("Cleaning up threads and tasks...")
    
    
    # Cancel the camera stream task
    try:
        if hasattr(server_data, 'ws_camera_stream_task') and server_data.ws_camera_stream_task:
            server_data.ws_camera_stream_task.cancel()
    except Exception as e:
        print(f"Error canceling camera task: {e}")

    # Stop the image processing thread
    try:
        if hasattr(image_proc_tsk_handler, 'stop'):
            image_proc_tsk_handler.stop()
            image_proc_tsk_handler.join(timeout=3) # 3-second timeout prevents freezing
    except Exception as e:
        print(f"Error stopping image thread: {e}")


def shutdown_raspberry_pi():
    """Cleans up the app and completely powers off the Raspberry Pi."""
    print("Initiating system shutdown...")
    cleanup_resources()
    
    # 'sudo poweroff' is the standard way to shut down a Raspberry Pi securely
    os.system("sudo poweroff")

def restart_python_program():
    """Cleans up the app and restarts the Python script within the same venv."""
    print("Restarting Python program...")
    cleanup_resources()
    
    # Force-close all inherited file descriptors/sockets (ports 5000, 8000, etc.)
    # 3 is the first non-standard descriptor. 65535 covers the maximum limit.
    try:
        os.closerange(3, 65535)
    except Exception as e:
        print(f"Error closing ports: {e}")
    
    # sys.executable ensures it uses the exact same venv Python binary
    # sys.argv passes along any command line arguments you originally used
    os.execv(sys.executable, [sys.executable] + sys.argv)

def check_app_ready():
    while not get_server_status():
        time.sleep(1)

if __name__ == "__main__":
    time.sleep(0.5)
    
    ## looking for program updates at startup
    check_for_updates_and_restart() 
    print("Starting Pest Rover...")

    
    ## initialize the cameras
    camera = CAMERA_DRIVER((1280, 720), raspCam=True, cameraID=0)

    ## initialize the server data structure
    server_data = SERVER_DATA_STRUCT()
    global_data = GLOBAL_DATA_STRUCT()

    ## load and sync settings from JSON file
    json_cfg = JsonConfigManager(global_settings, "saved_settings.json")
    json_cfg.load_and_sync()

    ## initialize the threads
    image_proc_tsk_handler = ImageProcessWorker(server_data)
    serial_handler = SerialComHandler()
    rover_controller = RoverControllerWorker(serial_handler, image_proc_tsk_handler)
    flask_app_thread = threading.Thread(target=run_web_app, daemon=True)

    ## start the threads
    serial_handler.start()
    image_proc_tsk_handler.start()
    rover_controller.start()
    flask_app_thread.start()
    
    ## wait for the web app to be ready before signaling the system is ready
    check_app_ready_th = threading.Thread(target=check_app_ready, daemon=True)
    check_app_ready_th.start()

    print("App Started!")

    try:
        asyncio.run(start_ws_server())
    
    except KeyboardInterrupt:
        print("Closing App")

        cleanup_resources()
        exit()


