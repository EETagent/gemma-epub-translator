<h3 align="center">Gemma Translator</h3>
<p align="center">
    <strong>Version: </strong> WIP
    <br />
    <br />
    <img src="https://img.shields.io/badge/macOS-purple.svg">
    <img src="https://img.shields.io/badge/llama.cpp🦙-yellow">
    <img src="https://img.shields.io/badge/Rust-🦀-blue.svg">
    <img src="https://img.shields.io/badge/Objective-C-gold.svg">
    <br />
    <br />
  </p>
</p>

<img width="1103" height="837" src="https://github.com/user-attachments/assets/606a7c8a-d0aa-4c11-8fe7-8fc3fbae743a" />

## Setup 📦

1. Clone the repository:
   ```bash
   git clone https://github.com/EETagent/gemma-translator.git
   cd gemma-translator
   ```

2. Download a model compatible with `llama.cpp` and place it in the `models/` directory (and update the path in the code if necessary):
- [translategemma-12b-it-GGUF](https://huggingface.co/mradermacher/translategemma-12b-it-GGUF)
- [translategemma-4b-it-GGUF](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF)


## Configure ⚙️

1. Update the configuration constants in the source (`src/translate.rs`) to match your model and settings:

```rust
// translategemma-4b-it-q8_0.gguf
const MODEL_FILE: &str = "translategemma-12b-it.Q4_K_M.gguf";
const CTX_SIZE: u32 = 2048 * 16;
const MAX_OUTPUT_TOKENS: usize = 512 * 2;
const SHORT_OUTPUT_TOKENS: usize = 16;
const SHORT_INPUT_TOKENS: usize = 4;
const MAX_SEQ_BATCH: usize = 8 * 2;
const TOP_K: i32 = 64;
const TOP_P: f32 = 0.95;
```

This takes about 19GB of RAM

## Build 🔨

1. Build the Rust backend:
   ```bash
   cargo build --release
   ```

2. Or bundle using `cargo-bundle` for macOS:
   ```bash
   cargo install cargo-bundle
   cargo bundle --release
   ```

## Run 🚀

1. Run the application:
   ```bash
   ./target/release/gemma-translator
   ```
