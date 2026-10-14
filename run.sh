#!/usr/bin/env bash
set -euo pipefail
camera_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$camera_root" -B "$camera_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$camera_root/build" -j4
exec "$camera_root/build/detect" "$@"
