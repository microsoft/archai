import subprocess


def spawn(cmd, env=None, cwd=None, check=False):
    out = subprocess.run(cmd, env=env, cwd=cwd, check=check,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    returncode = out.returncode
    stdout = out.stdout.decode("utf-8")
    stderr = out.stderr.decode("utf-8")
    return returncode, stdout, stderr


class Adb:
    def __init__(self):
        self.devices = []

    def get_devices(self):
        rc, stdout, stderr = spawn(["adb", "devices"])
        if rc != 0:
            raise Exception("Error: {}".format(stderr))

        result = []
        for line in stdout.splitlines():
            if not line.startswith("List of devices attached") and line:
                result.append(line.split('\t')[0])

        self.devices = result
        return result

    def ls(self, device, path):
        rc, stdout, stderr = spawn(["adb", "-s", device, "shell", "ls", path])
        if rc != 0:
            raise Exception("Error: {}".format(stderr))

        return stdout.splitlines()
