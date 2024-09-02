# %%
import os

import torch
import torch._inductor.config as inductor_config
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from onediffx import compile_pipe
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench

from onediff.utils.import_utils import is_nexfort_available

torch.set_default_device('cuda')
# RuntimeError: RuntimeError: Unsupported timesteps dtype: c10::BFloat16
# ref: https://github.com/siliconflow/onediff/issues/1066#issuecomment-2271523799
os.environ['NEXFORT_FUSE_TIMESTEP_EMBEDDING'] = '0'
os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
# %%
"""
    While using FLUX dev, is mandatory to provide the HF_TOKEN environment variable.
"""
# os.environ["HF_TOKEN"] = ""

model_id: str = "black-forest-labs/FLUX.1-dev" # "black-forest-labs/FLUX.1-schnell"
inductor_native = True
diffusion_steps = 20

pipe = FluxPipeline.from_pretrained(model_id, 
                                    torch_dtype=torch.bfloat16, 
                                    )
pipe.to("cuda")
# %%
"""
options = '{"mode": "O3"}' 
pipe.transformer = compile(pipe.transformer, backend="nexfort", options=options)
"""
"""
ref: https://github.com/siliconflow/onediff/blob/main/src/onediff/infer_compiler/README.md
compiler optimization options:
    - {"mode": "O2"}
    - {"mode": "O3", "memory_format": "channels_last"}
    - {"mode": "max-optimize:max-autotune:low-precision:cache-all"}
    - {"mode": "max-optimize:max-autotune:cudagraphs:low-precision:cache-all"}


| Mode | Description |
| - | - |
| `cache-all` | Cache all the compiled stuff to speed up reloading and recompiling. |
| `max-autotune` | Enable all the kernel autotuning options to find out the best kernels, this might slow down the compilation. |
| `max-optimize` | Enable the ***most*** extreme optimization strategies like the most aggressive fusion kernels to maximize the performance, this might slow down the compilation and require long autotuning. |
| `cudagraphs` | Enable CUDA Graphs to reduce CPU overhead. |
| `freezing` | Freezing will attempt to inline weights as constants in optimization and run constant folding and other optimizations on them. After freezing, weights can no longer be updated. |
| `low-precision` | Enable low precision mode. This will allow some math computations happen in low precision to speed up the overall performance. |

compiler_modes = collections.OrderedDict(
    {
        "max-optimize:max-autotune:low-precision": "This will deliver a good performance and adapt quickly to shape changes.",
        "max-optimize:max-autotune:low-precision:freezing:benchmark": "",
        "jit:disable-runtime-fusion:low-precision": "This compiles super quickly, but the performance might not be optimized very noticeably.",
        "jit:benchmark:low-precision:freezing:cudagraphs": "This compiles the model very quickly, but the performance might be not as good as `TorchInductor` optimized models.",
        "max-autotune:benchmark:low-precision:cudagraphs": "This is the most suggested combination of compiler modes. It will deliver a good balance between performance and compilation time.",
        "max-optimize:max-autotune:benchmark:low-precision:freezing:cudagraphs": "This is the most aggressive combination of compiler modes. It will deliver the best performance but might slow down the compilation significantly.",
    }
    )
"""
if is_nexfort_available():
    options = '{"mode": "max-optimize:max-autotune:benchmark:low-precision:freezing:cudagraphs"}'
    pipe = compile_pipe(pipe, backend="nexfort", options=options, fuse_qkv_projections=True)
if inductor_native:
    inductor_config.max_autotune_gemm_backends = "CUTLASS"
    inductor_config.max_autotune_gemm_search_space = "EXHAUSTIVE"

    pipe.vae = torch.compile(pipe.vae)
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
    pipe.transformer = torch.compile(pipe.transformer)
# %%
import time  # noqa: E402

prompt = "A cat holding a sign that says hello world"
st = time.time()
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=diffusion_steps,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
torch.cuda.synchronize()
et_fwd = time.time() - st
print(f"Time taken for forward pass: {et_fwd:.6f} s")
image.save("flux-schnell.png")
# %%
def get_flops_achieved(f):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        f()
    total_flops = flop_counter.get_total_flops()
    ms_per_iter = do_bench(f)
    iters_per_second = 1e3/ms_per_iter
    print(f"{iters_per_second * total_flops / 1e12} TF/s")


get_flops_achieved(lambda: pipe(
                "A tree in the forest",
                guidance_scale=0.0,
                num_inference_steps=diffusion_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
))
# %%
def benchmark_nexfort_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) * 1000 / iters, *f(*args, **kwargs)
# %%
time_nextfort_flux_fwd, _ = benchmark_nexfort_function(100,
                                                    pipe,
                                                    "A tree in the forest",
                                                    guidance_scale=0.0,
                                                    num_inference_steps=diffusion_steps,
                                                    max_sequence_length=256,
                                                    )
print(f"avg fwd time: {time_nextfort_flux_fwd / 1e6} s")
# %%
