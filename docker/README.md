# Using Docker to Run Archai

This folder contains tools for creating development and production environments that are secure and isolated from the host system, including Docker and gVisor.

## Docker

The Dockerfile can be used to build a development environment for running experiments. The `build_image.sh` and `run_container.sh` scripts can be used to build the Docker image and run the container, respectively:

```bash
bash build_image.sh
bash run_container.sh
```

## Docker and gVisor for Enhanced Security

[gVisor](https://gvisor.dev) is a runtime that provides an additional layer of security for containers by intercepting and monitoring runtime instructions before they reach the host system. Its primary goal is to enable the execution of untrusted workloads without compromising the security of other workloads or the underlying infrastructure.

To install the latest release of gVisor and use it as a Docker runtime:

Download and install gVisor:

```bash
(
  set -e
  ARCH=$(uname -m)
  URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
  wget ${URL}/runsc ${URL}/runsc.sha512 ${URL}/containerd-shim-runsc-v1 ${URL}/containerd-shim-runsc-v1.sha512
  sha512sum -c runsc.sha512 -c containerd-shim-runsc-v1.sha512
  rm -f *.sha512
  chmod a+rx runsc containerd-shim-runsc-v1
  sudo mv runsc containerd-shim-runsc-v1 /usr/local/bin
)
```

Set gVisor as the Docker runtime:

```bash
sudo /usr/local/bin/runsc install
sudo systemctl restart docker
```

To run the container with Docker and gVisor:

```bash
bash run_container_with_gvisor.sh
```