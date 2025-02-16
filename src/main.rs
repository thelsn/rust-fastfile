#[cfg(feature = "ai")]
mod document_parser;

#[cfg(feature = "ai")]
mod ai_search;

mod config;

use dashmap::DashMap;
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read},
    os::windows::fs::OpenOptionsExt,
    path::Path,
    sync::{mpsc::channel, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use walkdir::WalkDir;
use zstd::stream::{Decoder, Encoder};


const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x08000000;
const BUFFER_SIZE: usize = 32 * 1024 * 1024;
const RAYON_THREAD_COUNT: usize = 32;


#[derive(Serialize, Deserialize, Clone)]
struct FileEntry {
    path: String,
    lower_name: String,
}

#[derive(Serialize, Deserialize)]
struct FileIndex {
    base_root: String, // new field to record the initial root directory
    #[serde(skip)]
    files: DashMap<String, FileEntry>,
    sorted_entries: Vec<FileEntry>,
}


fn save_index(index: &FileIndex, path: &str) {
    let start = Instant::now();
    let file = File::create(path).expect("Failed to create index file");
    let writer = BufWriter::with_capacity(BUFFER_SIZE, file);
    let mut encoder = Encoder::new(writer, 3).expect("Failed to create zstd encoder");
    serde_json::to_writer(&mut encoder, index).expect("Failed to write index");
    encoder.finish().expect("Failed to finish compression");
    println!("save_index took {:?}", start.elapsed());
}

fn load_index(path: &str) -> Option<FileIndex> {
    let start = Instant::now();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
        .open(path)
        .ok()?;

    let mut decoder = Decoder::new(file).ok()?;
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).ok()?;
    let index: FileIndex = serde_json::from_slice(&decompressed).ok()?;

    println!("load_index took {:?}", start.elapsed());
    Some(index)
}

// Update build_index to record the provided root directory.
fn build_index(root: &str) -> FileIndex {
    let start = Instant::now();
    // Define ignore patterns.
    let ignore_patterns = ["node_modules", "System Volume Information", ".git", "target", "vendor"];
    let entries: Vec<FileEntry> = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy();
            // Check if the file's name contains any ignore pattern.
            let ignore = ignore_patterns.iter().any(|pat| name.contains(pat));
            !ignore && e.file_type().is_file()
        })
        .map(|e| FileEntry {
            path: e.path().to_string_lossy().to_string(),
            lower_name: e.file_name().to_string_lossy().to_lowercase(),
        })
        .collect();

    let mut sorted_entries = entries;
    sorted_entries.par_sort_unstable_by(|a, b| a.path.cmp(&b.path));

    let files = DashMap::with_capacity(sorted_entries.len());
    for entry in &sorted_entries {
        files.insert(entry.lower_name.clone(), entry.clone());
    }

    println!("build_index took {:?}", start.elapsed());
    FileIndex {
        base_root: root.to_string(),
        files,
        sorted_entries,
    }
}

// New helper: expand_index walks the new_root and adds file entries not yet in index.
fn expand_index(index: &mut FileIndex, new_root: &str) {
    println!("Expanding index for new root: {}", new_root);
    let ignore_patterns = ["node_modules", "System Volume Information", ".git", "target", "vendor"];
    // Walk new_root directory and ignore unwanted directories.
    for entry in WalkDir::new(new_root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy();
            let ignore = ignore_patterns.iter().any(|pat| name.contains(pat));
            !ignore && e.file_type().is_file()
        })
    {
        let path_str = entry.path().to_string_lossy().to_string();
        let lower = entry.file_name().to_string_lossy().to_lowercase();
        // Add file if not present
        if index.files.get(&lower).is_none() {
            let file_entry = FileEntry {
                path: path_str.clone(),
                lower_name: lower.clone(),
            };
            index.sorted_entries.push(file_entry.clone());
            index.files.insert(lower, file_entry);
        }
    }
    // Optional: update sorted_entries order
    index.sorted_entries.par_sort_unstable_by(|a, b| a.path.cmp(&b.path));
    // Update base_root to new_root once expanded
    index.base_root = new_root.to_string();
}

fn start_watcher(root: String, index_path: String, index: Arc<Mutex<FileIndex>>) {
    thread::spawn(move || {
        let (tx, rx) = channel();
        let mut watcher =
            RecommendedWatcher::new(tx, Config::default()).expect("Failed to create watcher");

        watcher
            .watch(Path::new(&root), RecursiveMode::Recursive)
            .expect("Failed to watch directory");

        let mut last_update = Instant::now();
        const UPDATE_COOLDOWN: Duration = Duration::from_secs(60);

        for res in rx {
            if let Ok(event) = res {
                if let EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) =
                    event.kind
                {
                    let now = Instant::now();
                    if now.duration_since(last_update) >= UPDATE_COOLDOWN {
                        last_update = now;
                        let mut index_guard = index.lock().unwrap();
                        let rebuild_start = Instant::now();
                        *index_guard = build_index(&root);
                        println!("Watcher rebuild_index took {:?}", rebuild_start.elapsed());
                        save_index(&index_guard, &index_path);
                        println!("Index updated due to file system change.");
                    }
                }
            }
        }
    });
}

fn search_index(index: &FileIndex, query: &str) -> Vec<String> {
    let start = Instant::now();
    let query_lower = query.to_lowercase();
    let mut found_files = Vec::new();

    if query_lower.starts_with('.') {
        // Extension search: check if the file path ends with the query extension.
        index.files.iter().for_each(|r| {
            if r.value().path.to_lowercase().ends_with(&query_lower) {
                println!("{}", r.value().path);
                found_files.push(r.value().path.clone());
            }
        });
    } else {
        // Regular search: match query among words.
        index.files.iter().for_each(|r| {
            let words: Vec<&str> = r.key().split(|c: char| !c.is_alphanumeric()).collect();
            if words.iter().any(|&word| word == query_lower) {
                println!("{}", r.value().path);
                found_files.push(r.value().path.clone());
            }
        });
    }
    
    println!("search_index took {:?}", start.elapsed());
    found_files
}

fn search_filesystem(root: &str, query: &str, ignore_list: &[String]) {
    let start = Instant::now();
    let query = query.to_lowercase();

    WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if entry.file_type().is_file() {
                    let path_str = entry.path().to_string_lossy().to_string();
                    if !ignore_list.contains(&path_str) {
                        let file_name = entry.file_name().to_string_lossy().to_lowercase();
                        // Split the file name into words using any non-alphanumeric character as a delimiter
                        let words: Vec<&str> =
                            file_name.split(|c: char| !c.is_alphanumeric()).collect();
                        // Check if any of the words in the file name match the query
                        if words.iter().any(|&word| word == query) {
                            println!("{}", path_str);
                            return Some(());
                        }
                    }
                }
            }
            None
        })
        .count();

    println!("search_filesystem took {:?}", start.elapsed());
}

fn search_folder_fs(root: &str, query: &str) {
    let start = Instant::now();
    let query = query.to_lowercase();

    WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if entry.file_type().is_dir() {
                    let folder_name = entry.file_name().to_string_lossy().to_lowercase();
                    let words: Vec<&str> =
                        folder_name.split(|c: char| !c.is_alphanumeric()).collect();
                    if words.iter().any(|&word| word == query) {
                        println!("{}", entry.path().to_string_lossy());
                        return Some(());
                    }
                }
            }
            None
        })
        .count();

    println!("search_folder_fs took {:?}", start.elapsed());
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(RAYON_THREAD_COUNT)
        .build_global()
        .unwrap();

    let overall_start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} (-f | -d | -a) <search_query> [root_dir]",
            args[0]
        );
        return;
    }

    let search_type = &args[1];
    let query = args[2].to_lowercase();
    // Here you can use the new root directory from config
    // For demonstration, we'll use a new_root variable. Replace as needed.
    let new_root = "C:\\".to_string();
    let index_file = "index.filemap";

    match search_type.as_str() {
        "-f" => {
            let load_start = Instant::now();
            let mut index = match load_index(index_file) {
                Some(mut idx) => {
                    println!("Loaded index from file.");
                    // Repopulate the DashMap from sorted_entries in parallel
                    let files = DashMap::with_capacity(idx.sorted_entries.len());
                    idx.sorted_entries.par_iter().for_each(|entry| {
                        files.insert(entry.lower_name.clone(), entry.clone());
                    });
                    idx.files = files;
                    idx
                }
                None => {
                    println!("No existing index found. Building from scratch...");
                    let build_start = Instant::now();
                    let idx = build_index(&new_root);
                    println!("Building index took {:?}", build_start.elapsed());
                    let save_start = Instant::now();
                    save_index(&idx, index_file);
                    println!("Saving index took {:?}", save_start.elapsed());
                    idx
                }
            };
            println!("Load/Build index took {:?}", load_start.elapsed());

            // If new_root differs from the stored base_root, expand index:
            if !new_root.to_lowercase().starts_with(&index.base_root.to_lowercase())
                && !index.base_root.to_lowercase().starts_with(&new_root.to_lowercase())
            {
                // Non-overlapping roots ? For simplicity, rebuild the index.
                println!("New root is different from indexed root. Rebuilding index.");
                index = build_index(&new_root);
            } else if new_root.to_lowercase() != index.base_root.to_lowercase() {
                expand_index(&mut index, &new_root);
            }
            save_index(&index, index_file);

            // ...existing file search using the updated index...
            let index_arc = Arc::new(Mutex::new(index));
            start_watcher(new_root.clone(), index_file.to_string(), Arc::clone(&index_arc));

            let search_start = Instant::now();
            let found_files = search_index(&index_arc.lock().unwrap(), &query);
            let index_search_time = search_start.elapsed();

            if found_files.is_empty() {
                println!(
                    "No results in index. Running filesystem search and updating index concurrently..."
                );
                let root_clone_for_search = new_root.clone();
                let query_clone = query.clone();
                let fs_search_handle = thread::spawn(move || {
                    search_filesystem(&root_clone_for_search, &query_clone, &found_files);
                });

                let root_clone_for_build = new_root.clone();
                let build_handle = thread::spawn(move || {
                    let update_start = Instant::now();
                    let new_idx = build_index(&root_clone_for_build);
                    println!("Index rebuild took {:?}", update_start.elapsed());
                    new_idx
                });

                fs_search_handle.join().unwrap();
                let new_index = build_handle.join().unwrap();
                {
                    let mut index_guard = index_arc.lock().unwrap();
                    *index_guard = new_index;
                }
                save_index(&index_arc.lock().unwrap(), index_file);
            } else {
                println!("Results found in the index, skipping filesystem scan and index rebuild.");
            }

            println!("\nSearch Statistics:");
            println!("Index search time: {:.2?}", index_search_time);
            println!("Total runtime: {:.2?}", overall_start.elapsed());
        }
        "-d" => {
            let search_start = Instant::now();
            search_folder_fs(&new_root, &query);
            println!("Folder search took {:?}", search_start.elapsed());
            println!("Total runtime: {:.2?}", overall_start.elapsed());
        }
        "-a" => {
            #[cfg(feature = "ai")]
            {
                // If a new location is specified as a fourth argument, use it.
                let ai_root = if args.len() >= 4 {
                    args[3].clone()
                } else {
                    new_root.clone()
                };
                // Load configuration from file.
                let config = crate::config::Config::load("config.toml")
                    .expect("Failed to load config.toml");
                println!("Running AI semantic search for query: {} in {}", query, ai_root);
                if let Err(e) = ai_search::run_ai_search(&query, &ai_root, index_file, &config) {
                    eprintln!("AI search error: {}", e);
                }
            }
            #[cfg(not(feature = "ai"))]
            {
                eprintln!("AI search feature is not enabled. Rebuild with --features ai");
            }
        }
        _ => {
            eprintln!(
                "Usage: {} (-f | -d | -a) <search_query> [root_dir]",
                args[0]
            );
        }
    }
}
