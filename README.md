# FastFile Search

A file search tool that fast. Multiple search modes. Regular file search, folder search, and AI semantic search.

## Features

- **Fast File Search (-f)**: Indexed file search with real-time filesystem monitoring
- **Directory Search (-d)**: Search for directories/folders
- **AI Semantic Search (-a)**: Find files based on meaning with BERT embeddings
- **GPU Acceleration**: Uses CUDA if available for AI
- **Parallel Processing**: Multiple threads for faster search
- **Compressed Index**: Compressed index for quick searches

## Usage

```
fastfile_search.exe <mode> <query> [root_dir]

Modes:
  -f    Regular file search (uses index)
  -d    Directory/folder search
  -a    AI semantic search
```

## Building

Basic build (file and directory search only):
```bash
cargo build --release
```

Build with AI search support:
```bash
cargo build --release --features ai
```

## Requirements

- Basic: Standard Rust toolchain
- AI Search (Optional): 
  - CUDA-capable GPU and CUDA 11.8 *(if you want it to be fast, but you could do it on your cpu)*
  - Torch 2.4.0 *(no other version will work has to be 2.4.0)*
