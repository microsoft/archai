import os
import sys
import json
import subprocess

home = os.getenv("HOME")
experiments = f"{home}/snpe/experiment"
script_dir = os.path.dirname(os.path.abspath(__file__))
map_file = os.path.join(script_dir, "map.txt")
command = f'{script_dir}/../azure/loop.sh'


def run(cmd):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr = p.communicate()
    return (stdout.strip(), stderr.strip())


def find_screens():
    screens = {}
    stdout, stderr = run(["screen", "-list"])
    if stderr != "":
        print("screen -list failed: ")
        print(stderr)
        sys.exit(1)
    for line in stdout.splitlines():
        parts = line.strip().split('\t')
        if len(parts) > 1:
            id = parts[0]
            parts = id.split(".")
            if len(parts) == 2:
                proc, device = parts
                screens[device] = proc
            else:
                print("### skipping unrecognized screen name: {id}")
    return screens


def find_devices():
    devices = []
    stdout, stderr = run(["adb", "devices"])
    if stderr != "":
        print("adb devices failed: ")
        print(stderr)
        sys.exit(1)
    for line in stdout.splitlines():
        parts = line.split('\t')
        if len(parts) == 2 and parts[1] == 'device':
            devices += [parts[0]]
    return devices


def load_map():
    map = {}
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            map = json.load(f)
    return map


def save_map(map):
    with open(map_file, "w") as f:
        json.dump(map, f, indent=2)
    return map


def allocate_folder(map, id):
    next = 1
    inverse_map = {v: k for k, v in map.items()}
    while True:
        folder = f"{experiments}{next}"
        if folder not in inverse_map:
            map[id] = folder
            save_map(map)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            return folder
        else:
            next += 1


def read_lock(folder):
    lock_file = os.path.join(folder, "lock.txt")
    if os.path.exists(lock_file):
        return open(lock_file).read().strip()
    return None


def main():
    devices = find_devices()
    if len(devices) == 0:
        print("### Found no Qualcomm Devices using `adb devices`")
        sys.exit(1)

    screens = find_screens()
    map = load_map()
    print("# Found the following Qualcomm Devices using `adb devices`:")
    for id in find_devices():
        if id in map:
            folder = map[id]
        else:
            folder = allocate_folder(map, id)
        lock = read_lock(folder)
        print(f"Device {id}, mapped to folder {folder}")
        if id in screens:
            proc = screens[id]
            print(f"  Screen is running: {proc}.{id}")
        elif lock:
            print(f"  Locked by: {lock}")
        else:
            print(f"  Please run:  screen -dmS {id} {command} --device {id} --no_quantization " +
                  f"--cleanup_stale_pods 3600 --working {folder}")


if __name__ == '__main__':
    main()
