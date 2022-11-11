# Running Archai with Docker

This folder provides tools for creating secure development and production environments, such as Docker and gVisor.

## Docker

The `Dockerfile` provides a development environment to run experiments. Additionally, `build_image.sh` and `run_container.sh` provides scripts to build the image and run the container, respectively:

```bash
bash build_image.sh
bash run_container.sh
```

## Docker + gVisor (safe environment)

[gVisor](https://gvisor.dev) implements an additional security layer for containers by intercepting and monitoring runtime instructions before they reach the underlying host. Its main goal is to allow untrusted workloads without compromising other workloads or underlying infrastructure.

The following steps describe how to download and install the latest release:

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

Additionally, one needs to install gVisor as a Docker runtime:

```bash
sudo /usr/local/bin/runsc install
sudo systemctl restart docker
```

Finally, the container (Docker + gVisor) can be run as follows:

```bash
bash run_container_with_gvisor.sh
```