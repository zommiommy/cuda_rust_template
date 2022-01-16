# cuda_rust_template
An example of fully rust CUDA code with as little dependancies as possible.

To setup and run:
```bash
# add the toolchain with:
rustup target add nvptx64-nvidia-cuda

# install the linker
cargo install ptx-linker -f --version ">= 0.9"

# go in the cpu code folder
cd cpu_code

# Compile everything and run
cargo run --release
```

This is a really bare-bone example and the goal of this repository is to give an working example with as little magical wrappers as possible.