# Makefile — verify and build YOUR GEMM kernel (src/kernel.cpp).
#
#   make csim      C simulation of src/kernel_tb.cpp (all 9 graded cases)
#   make csynth    HLS synthesis -> build/hls/gemm.xo  (+ report: II, timing, resources)
#   make cosim     RTL co-simulation on tiny shapes (runs csynth first if needed)
#   make hw_emu    emulation bitstream -> build/hw_emu/gemm.xclbin   (~5 min)
#   make hw        hardware bitstream  -> build/hw/gemm.xclbin       (~1 h)
#   make submit    upload build/hw/gemm.xclbin to the grader (needs HW_TOKEN)
#   make clean
#
# On RCE never run these on the login node: quick steps in `sinteractive -c 8 -A cs2401`, long ones via ./slurm.sh <target> (see README).

SHELL  := /bin/bash
ENV    := source $(CURDIR)/env.sh >/dev/null;
BUILD  := build
HLS    := $(BUILD)/hls
XO     := $(HLS)/gemm.xo
REPORT := $(HLS)/hls/syn/report/gemm_csynth.rpt

.PHONY: csim csynth cosim hw_emu hw submit clean

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
	$(ENV) cd $(BUILD)/$* && v++ -l -t $* --platform $$PLATFORM $(CURDIR)/$(XO) -o gemm.xclbin
	@echo "bitstream: $@"

submit:
	bash scripts/hw-submit.sh $(BUILD)/hw/gemm.xclbin

clean:
	rm -rf $(BUILD) _x .Xil *.log *.jou
