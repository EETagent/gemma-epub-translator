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
