# FastFile Search

A file search tool that fast. Multiple search modes. Regular file search, folder search, and AI semantic search.

## Features

**The AI bit of this sucks like crazy i wouldnt recommnd using it on any large file system or even a small one because well it takes 15 min for 2477283 files on my 4070**

- **Fast File Search (-f)**: Indexed file search with real-time filesystem monitoring
- **Directory Search (-d)**: Search for directories/folders
- **AI Semantic Search (-a)**: Find files based on meaning with BERT embeddings
- **GPU Acceleration**: Uses CUDA if available for AI
- **Live Updates**: Shows results as found
- **Parallel Processing**: Multiple threads for faster search
- **Compressed Index**: Compressed index for quick searches


## Usage

```
fastfile_search.exe <mode> <query> [root_dir]

Modes:
  -f    Regular file search (uses index)
  -d    Directory/folder search
  -a    AI semantic search *not recommended*

```
