# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import subprocess
from threading import Thread, Lock


class Shell:
    def __init__(self):
        self.output = ''
        self.lock = Lock()
        self.verbose = False

    def run(self, cwd, command, print_output=True):
        self.output = ''
        self.verbose = print_output
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                              universal_newlines=True, cwd=cwd, shell=True) as proc:

            stdout_thread = Thread(target=self.logstream, args=(proc.stdout,))
            stderr_thread = Thread(target=self.logstream, args=(proc.stderr,))

            stdout_thread.start()
            stderr_thread.start()

            while stdout_thread.is_alive() or stderr_thread.is_alive():
                pass

            proc.wait()

            if proc.returncode:
                words = command.split(' ')
                print("### command {} failed with error code {}".format(words[0], proc.returncode))
                raise Exception(self.output)
            return self.output

    def logstream(self, stream):
        try:
            while True:
                out = stream.readline()
                if out:
                    self.log(out)
                else:
                    break
        except Exception as ex:
            msg = "### Exception: {}".format(ex)
            self.log(msg)

    def log(self, msg):
        self.lock.acquire()
        try:
            msg = msg.rstrip('\r\n')
            self.output += msg + '\r\n'
            if self.verbose:
                print(msg)
        finally:
            self.lock.release()
