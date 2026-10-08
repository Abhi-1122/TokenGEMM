# Makefile — verify and build the GEMM kernel (src/kernel.cpp).
#
#   make csim      C simulation of src/kernel_tb.cpp (all 9 test shapes)
#   make csynth    HLS synthesis -> build/hls/gemm.xo  (+ report: II, timing, resources)
#   make cosim     RTL co-simulation on tiny shapes (runs csynth first if needed)
#   make hw_emu    emulation bitstream -> build/hw_emu/gemm.xclbin   (~5 min)
#   make hw        hardware bitstream  -> build/hw/gemm.xclbin       (~3.5 h)
#   make clean
#
# On a Slurm cluster, run long steps (hw, hw_emu) as a job: ./slurm.sh <target>.

SHELL  := /bin/bash
ENV    := source $(CURDIR)/env.sh >/dev/null;
BUILD  := build
HLS    := $(BUILD)/hls
XO     := $(HLS)/gemm.xo
REPORT := $(HLS)/hls/syn/report/gemm_csynth.rpt

.PHONY: csim csynth cosim hw_emu hw clean

csim:
	$(ENV) vitis-run --mode hls --csim --config hls_config.cfg --work_dir $(HLS)

csynth: $(XO)
$(XO): src/kernel.cpp hls_config.cfg
	$(ENV) v++ -c --mode hls --config hls_config.cfg --work_dir $(HLS)
	@echo "csynth report: $(REPORT)"

cosim: $(XO)
	$(ENV) vitis-run --mode hls --cosim --config hls_config.cfg --work_dir $(HLS)

hw_emu: $(BUILD)/hw_emu/gemm.xclbin
hw:     $(BUILD)/hw/gemm.xclbin
$(BUILD)/%/gemm.xclbin: $(XO)
	mkdir -p $(BUILD)/$*
	$(ENV) cd $(BUILD)/$* && v++ -l -t $* --platform $$PLATFORM --config $(CURDIR)/connectivity.cfg $(CURDIR)/$(XO) -o gemm.xclbin
	@echo "bitstream: $@"

clean:
	rm -rf $(BUILD) _x .Xil *.log *.jou
