# Windows / CUDA 11.8 / V100 (sm_70) build

Branch `win-cuda118` of the `feature/v100-moe` fork, for machines whose NVIDIA driver tops out at CUDA 11.x (tested on driver 472.12, Tesla V100 16GB).

## Changes

- `S01` CUDA standard 17 for nvcc 11.8 (nvcc 11.8 accepts at most C++17); host C++ stays at 20.
- `S02` `cudaGraphInstantiate` rewritten from the CUDA 12 3-argument form to the CUDA 11.8 5-argument form (8 call sites).

## Toolchain

- CUDA 11.8 (`nvcc`), MSVC 14.34 (VS 2022 Build Tools), Ninja, CMake
- `CMAKE_CUDA_ARCHITECTURES=70`, `CMAKE_BUILD_TYPE=Release`

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.34
cmake -S . -B ..\build-strata -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER=C:\path\to\cuda-11.8\bin\nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build ..\build-strata
ctest --test-dir ..\build-strata
```

ctest: 20/20 pass.

## Measured (Qwen3.8-Flash-Next, ISTA-DASLab GSQ-RCO Q2_0, V100 16GB + DDR4 host)

`--expert-cache 7500 --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp <q2_0 rt> --max-context 8192`, Swift expert profile:

| | tok/s |
|---|---|
| decode | 51.0 / 53.0 / 51.8 |
| prefill (2047 tokens) | 541 – 547 |

At `--max-context 131072 --kv int8` the largest expert cache that starts and serves is 6400 slots (VRAM 16.19 / 16.26 GB). A 30k-token needle test answered correctly.

The MTP runtime pack was built with `STRATA_GGUF_PY` pointing at a llama.cpp `gguf-py` checkout.
