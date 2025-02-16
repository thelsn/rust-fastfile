use dashmap::DashMap;
use indicatif::{ProgressBar, ProgressStyle};
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::prelude::*;
use rust_bert::{
    pipelines::{
        common::ModelType,
        sentence_embeddings::{SentenceEmbeddingsConfig, SentenceEmbeddingsModel},
    },
    resources::{RemoteResource, ResourceProvider},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, Metadata, OpenOptions},
    io::{BufWriter, Read},
    os::windows::fs::OpenOptionsExt,
    path::Path,
    sync::{mpsc::channel, Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime},
};
use tch::Device;
use walkdir::WalkDir;
use zstd::stream::{Decoder, Encoder};

const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x08000000;
const BUFFER_SIZE: usize = 32 * 1024 * 1024;
const RAYON_THREAD_COUNT: usize = 32;
const SEMANTIC_BATCH_SIZE: usize = 500; 
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;
const TOP_N_RESULTS: usize = 10; 

#[derive(Serialize, Deserialize, Clone)]
struct FileEntry {
    path: String,
    lower_name: String,
}

#[derive(Serialize, Deserialize)]
struct FileIndex {
    #[serde(skip)]
    files: DashMap<String, FileEntry>,
    sorted_entries: Vec<FileEntry>,
}

fn process_semantic_batch(
    model: &SentenceEmbeddingsModel,
    query_embedding: &[f32],
    file_metadata: &[(String, String)],
    results: &Arc<Mutex<Vec<(String, f32)>>>,
) {
    let metadata_texts: Vec<&String> = file_metadata.iter().map(|(_, content)| content).collect();
    let embeddings = model.encode(&metadata_texts).unwrap();

    let mut batch_results: Vec<(String, f32)> = file_metadata
        .iter()
        .zip(embeddings.iter())
        .map(|((file_path, _), emb)| {
            let similarity = cosine_similarity(query_embedding, emb);
            (file_path.clone(), similarity)
        })
        .collect();

    // Sort the batch results and extend the global results
    batch_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut global_results = results.lock().unwrap();
    global_results.extend(batch_results);
    global_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    global_results.truncate(TOP_N_RESULTS);
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

fn build_index(root: &str) -> FileIndex {
    let start = Instant::now();
    let entries: Vec<FileEntry> = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy();
            !name.contains("node_modules")
                && !name.contains("System Volume Information")
                && !name.contains(".git")
                && e.file_type().is_file()
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
        files,
        sorted_entries,
    }
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
    let query = query.to_lowercase();
    let mut found_files = Vec::new();

    index.files.iter().for_each(|r| {
        // Split the key into words using any non-alphanumeric character as a delimiter
        let words: Vec<&str> = r.key().split(|c: char| !c.is_alphanumeric()).collect();
        // Check if any of the words in the key match the query
        if words.iter().any(|&word| word == query) {
            println!("{}", r.value().path);
            found_files.push(r.value().path.clone());
        }
    });

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

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot: f32 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    dot / (norm1 * norm2)
}

fn get_metadata_string(metadata: &Metadata, path: &str) -> String {
    let modified_time: SystemTime = metadata.modified().unwrap();
    format!(
        "Path: {}\nSize: {}\nModified: {:?}",
        path,
        metadata.len(),
        modified_time
    )
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
            "Usage: {} (-f | -d | -v | -a) <search_query> [root_dir]",
            args[0]
        );
        return;
    }

    let search_type = &args[1];
    let query = args[2].to_lowercase();
    let root_dir = "C:\\".to_string();
    let index_file = "index.filemap";

    if search_type == "-f" {
        // Load index
        let load_start = Instant::now();
        let index = match load_index(index_file) {
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
                let idx = build_index(&root_dir);
                println!("Building index took {:?}", build_start.elapsed());
                let save_start = Instant::now();
                save_index(&idx, index_file);
                println!("Saving index took {:?}", save_start.elapsed());
                idx
            }
        };
        println!("Load/Build index took {:?}", load_start.elapsed());

        let index_arc = Arc::new(Mutex::new(index));

        start_watcher(
            root_dir.to_string(),
            index_file.to_string(),
            Arc::clone(&index_arc),
        );

        let search_start = Instant::now();
        let found_files = search_index(&index_arc.lock().unwrap(), &query);
        let index_search_time = search_start.elapsed();

        if found_files.is_empty() {
            println!(
                "No results in index. Running filesystem search and updating index concurrently..."
            );

            let root_clone_for_search = root_dir.to_string();
            let query_clone = query.clone();
            let fs_search_handle = thread::spawn(move || {
                search_filesystem(&root_clone_for_search, &query_clone, &found_files);
            });

            let root_clone_for_build = root_dir.to_string();
            let build_handle = thread::spawn(move || {
                let update_start = Instant::now();
                let new_index = build_index(&root_clone_for_build);
                println!("Index rebuild took {:?}", update_start.elapsed());
                new_index
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
    } else if search_type == "-d" {
        let search_start = Instant::now();
        search_folder_fs(&root_dir, &query);
        println!("Folder search took {:?}", search_start.elapsed());
        println!("Total runtime: {:.2?}", overall_start.elapsed());
    } else if search_type == "-a" {
        println!("Running AI semantic search for query: {}", query);

        let load_start = Instant::now();
        let index = match load_index(index_file) {
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
                let idx = build_index(&root_dir);
                println!("Building index took {:?}", build_start.elapsed());
                let save_start = Instant::now();
                save_index(&idx, index_file);
                println!("Saving index took {:?}", save_start.elapsed());
                idx
            }
        };
        println!("Load/Build index took {:?}", load_start.elapsed());

        let device = Device::cuda_if_available();
        println!("Using device: {:?}", device);

        const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
        const MODEL_DIR: &str = "./models";

        std::fs::create_dir_all(MODEL_DIR).unwrap();

        let config = SentenceEmbeddingsConfig {
            modules_config_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/modules.json",
            ))) as Box<dyn ResourceProvider + Send>,
            transformer_config_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/config.json",
            ))) as Box<dyn ResourceProvider + Send>,
            transformer_weights_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/pytorch_model.bin",
            ))) as Box<dyn ResourceProvider + Send>,
            dense_config_resource: Some(Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/dense_config.json",
            ))) as Box<dyn ResourceProvider + Send>),
            dense_weights_resource: Some(Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/dense_model.bin",
            ))) as Box<dyn ResourceProvider + Send>),
            sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/sentence_bert_config.json",
            ))) as Box<dyn ResourceProvider + Send>,
            tokenizer_config_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/tokenizer_config.json",
            ))) as Box<dyn ResourceProvider + Send>,
            tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/vocab.txt",
            ))) as Box<dyn ResourceProvider + Send>,
            transformer_type: ModelType::Bert,
            pooling_config_resource: Box::new(RemoteResource::from_pretrained((
                MODEL_ID,
                "./models/pooling_config.json",
            ))) as Box<dyn ResourceProvider + Send>,
            device,
            kind: None,
            tokenizer_merges_resource: None,
        };

        let model_start = Instant::now();
        let model = SentenceEmbeddingsModel::new(config).unwrap();
        println!("Model loading took {:?}", model_start.elapsed());

        let query_encode_start = Instant::now();
        let query_embedding = model.encode(&[query.clone()]).unwrap()[0].clone();
        println!("Query encoding took {:?}", query_encode_start.elapsed());

        let file_paths: Vec<String> = index
            .sorted_entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();

        println!("Fetching file metadata...");
        let metadata_start = Instant::now();
        let file_metadata: Vec<(String, String)> = file_paths
            .par_iter()
            .filter_map(|file_path| {
                let metadata = std::fs::metadata(file_path).ok()?;
                if metadata.len() > MAX_FILE_SIZE {
                    return None;
                }
                let metadata_string = get_metadata_string(&metadata, file_path);
                Some((file_path.clone(), metadata_string))
            })
            .collect();
        println!("Fetching metadata took {:?}", metadata_start.elapsed());

        println!(
            "Processing {} files in batches of {}",
            file_metadata.len(),
            SEMANTIC_BATCH_SIZE
        );

        let pb = ProgressBar::new(file_metadata.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} (ETA: {eta})",
                )
                .unwrap()
                .progress_chars("##-"),
        );

        let all_results: Arc<Mutex<Vec<(String, f32)>>> = Arc::new(Mutex::new(Vec::new()));

        let batch_start = Instant::now();
        file_metadata.chunks(SEMANTIC_BATCH_SIZE).for_each(|chunk| {
            process_semantic_batch(&model, &query_embedding, chunk, &all_results);

            // Print top results periodically
            let top_results = all_results.lock().unwrap();
            println!("\nTop results:");
            for (file, score) in top_results.iter().take(TOP_N_RESULTS) {
                println!("{} (score: {:.4})", file, score);
            }
        });
        pb.finish_with_message("Done processing files.");
        println!("\nBatch processing took {:?}", batch_start.elapsed());

        let sort_start = Instant::now();
        let mut final_results = all_results.lock().unwrap().clone();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Sorting results took {:?}", sort_start.elapsed());

        println!("\nTop results:");
        for (file, score) in final_results.iter().take(TOP_N_RESULTS) {
            println!("{} (score: {:.4})", file, score);
        }
    }
}
