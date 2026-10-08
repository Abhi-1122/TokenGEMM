#!/usr/bin/env bash
# env.sh — put the U50 / Vitis 2025.1 toolchain + XRT on PATH for the hardware
# flow: building an .xclbin on the RCE cluster, and running it on the U50 host.
#   source ./env.sh      (the Makefile does this for you)
#
# Paths are for the RCE cluster, where /home is a shared mount visible on every build
# node and on the U50 host. On another machine, this is the only file to edit.

# Vitis HLS + v++ (2025.1 is the version validated for the U50 on this cluster).
source /home/Xilinx/2025.1/Vitis/settings64.sh

# XRT — needed to compile the host and to run hw/hw_emu (harmless for a pure build).
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true

# The U50 build platform (.xpfm) is on the shared NFS copy, not /opt/xilinx/platforms.
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms:/home/Xilinx/opt/xilinx/platforms
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
export PART=xcu50-fsvh2104-2-e

# FLEXlm license (needed for v++ hw/hw_emu; csim/csynth/cosim work without it).
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-2100@rce.iiit.ac.in}"
