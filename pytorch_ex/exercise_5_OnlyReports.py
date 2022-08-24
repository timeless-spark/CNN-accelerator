import sys
sys.path.insert(6, 'workspace/finn/src')
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil

model_file = "FashionMNISTbrevitas.onnx"

estimates_output_dir = "output_estimates_only"

#Delete previous run results if exist
if os.path.exists(estimates_output_dir):
    shutil.rmtree(estimates_output_dir)
    print("Previous run results deleted!")


cfg_estimates = build.DataflowBuildConfig(
    output_dir          = estimates_output_dir,
    mvau_wwidth_max     = 80,
    target_fps          = 1000000, #target inference performance in frames per second. 
                                   #Note that target may not be achievable due to specific 
                                   #layer constraints, or due to resource limitations of the FPGA. 
    synth_clk_period_ns = 10.0,    #Target clock frequency (in nanoseconds) for Vivado synthesis
    #fpga_part           = "xc7z020clg400-1",
    board               = "Pynq-zcu104",
    steps               = build_cfg.estimate_only_dataflow_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ]
)

%%time
build.build_dataflow_cfg(model_file, cfg_estimates)