# This script connects local adb devices to Azure IOT hub and
# implements some method calls to control the device.
# This assumes you have the Android ADB tool installed, and
# the SNPE SDK in an SNPE_ROOT, and the Olive2 SDK.

from utils import Adb, spawn
import json
import platform
import time
from client import SnpeClient


class DeviceManager:
    def __init__(self):
        config = json.load(open(".azureml/config.json"))
        self.subscription = config['subscription_id']
        self.iot_resource_group = config['iot_resource_group']
        self.iot_hub_name = config['iot_hub_name']
        self.location = config['location']

    def register(self):

        if platform.system() == 'Windows':
            az = 'az.cmd'
        else:
            az = 'az'


        rc, stdout, stderr = spawn([az, 'iot', 'hub', 'connection-string', 'show',
            '--hub-name',  self.iot_hub_name, '--resource-group', self.iot_resource_group])

        if rc != 0:
            raise Exception(stderr)

        config = json.loads(stdout)
        iot_hub_connection_string = config['connectionString']

        adb = Adb()
        devices = adb.get_devices()

        for device in devices:
            print(f"Checking Device Twin for android device {device}...")
            rc, stdout, stderr = spawn([az, 'iot', 'hub', 'device-identity', 'show',
            '--device-id', device,
            '--hub-name',  self.iot_hub_name, '--resource-group', self.iot_resource_group])

            if rc != 0:
                if 'DeviceNotFound' in stderr:

                    print(f"Creating Device Twin for android device {device}...")

                    rc, stdout, stderr = spawn([az, 'iot', 'hub', 'device-identity', 'create',
                        '--device-id', device,
                        '--hub-name',  self.iot_hub_name, '--resource-group', self.iot_resource_group])

                    if rc != 0:
                        raise Exception(stderr)

                else:
                    raise Exception(stderr)

        return devices



# Run the handler for each device attached to this local machine.
mgr = DeviceManager()
devices = mgr.register()
clients = []
for device in devices:
    client = SnpeClient(device, mgr.iot_hub_name, mgr.iot_resource_group)
    if client.connect():
        client.start_thread()
        clients.append(client)

while len(clients):
    for c in list(clients):
        if not c.connected:
            clients.remove(c)
    time.sleep(1)

print("all clients have terminated")