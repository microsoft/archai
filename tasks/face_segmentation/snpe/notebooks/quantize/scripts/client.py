from azure.iot.device import IoTHubDeviceClient, MethodResponse
from utils import Adb, spawn
import json
import platform
import time
from threading import Thread


class SnpeClient:
    def __init__(self, device, iot_hub_name, iot_resource_group):
        self.device = device
        self.iot_hub_name = iot_hub_name
        self.iot_resource_group = iot_resource_group
        self.adb = Adb()
        self.connected = False

    def connect(self):
        if platform.system() == 'Windows':
            az = 'az.cmd'
        else:
            az = 'az'

        rc, stdout, stderr = spawn([az, 'iot', 'hub', 'device-identity', 'connection-string', 'show',
            '--device-id', self.device, '--hub-name',  self.iot_hub_name, '--resource-group', self.iot_resource_group])

        if rc != 0:
            print(stderr)
            self.connected = False

        config = json.loads(stdout)
        self.connection_string = config['connectionString']

        # Instantiate the client
        self.client = IoTHubDeviceClient.create_from_connection_string(self.connection_string)

        try:
            # Attach the handler to the client
            self.client.on_method_request_received = self.method_request_handler
            self.connected = True
            return True
        except:
            # In the event of failure, clean up
            self.client.shutdown()
            self.connected = False
            return False

    def start_thread(self):
        self.stdin_thread = Thread(target=self.run, args=())
        self.stdin_thread.daemon = True
        self.stdin_thread.start()

    def run(self):
        print(f"running thread for device {self.device}...")
        while self.connected:
            time.sleep(1)

        # Define the handler for method requests
    def method_request_handler(self, method_request):
        if method_request.name == "ls":
            # run an adb command on the device.
            path = '/'
            if 'path' in method_request.payload:
                path = method_request.payload['path']
            print(f"device {self.device} is running ls {path}")
            result = self.adb.ls(self.device, path)

            # Create a method response indicating the method request was resolved
            resp_status = 200
            resp_payload = {"Response": result}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)

        elif method_request.name == "shutdown":
            print(f"device {self.device} is shutting down...")
            self.connected = False
            # Create a method response indicating the method request was resolved
            resp_status = 200
            resp_payload = {"Response": "Shutting down device " + self.device}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)

        else:
            # Create a method response indicating the method request was for an unknown method
            resp_status = 404
            resp_payload = {"Response": "Unknown method"}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)

        # Send the method response
        self.client.send_method_response(method_response)
