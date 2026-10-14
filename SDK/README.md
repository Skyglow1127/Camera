# Orbbec SDK

This folder contains only the files required to build and run the camera programs:

- `include/`: C and C++ headers
- `lib/`: runtime shared libraries
- `config/`: SDK runtime configuration and reference documentation
- `udev/`: Linux USB device rule and its installation script
- `LICENSE.txt`: SDK license

The project-level `CMakeLists.txt` resolves this folder through a relative path. No SDK path needs to be configured manually.

On a new Linux machine, install the USB rule once if the camera is not accessible:

```bash
sudo ./SDK/udev/install_udev_rules.sh
```
