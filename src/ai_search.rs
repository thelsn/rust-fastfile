use crate::{load_index, build_index, save_index};
use crate::config::Config;
use indicatif::{ProgressBar, ProgressStyle};
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsModel, SentenceEmbeddingsConfig};
use rust_bert::resources::RemoteResource;
use rust_bert::pipelines::common::ModelType;
use std::time::Instant;
use tch::Device;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use crate::document_parser::{self, DocumentContent};

// Removed hardcoded constants; values will come from config.
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot: f32 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    dot / (norm1 * norm2)
}

// Updated run_ai_search to accept a config reference.
pub fn run_ai_search(
    query: &str,
    root_dir: &str,
    index_file: &str,
    config: &Config
) -> Result<(), Box<dyn std::error::Error>> {
    let load_start = Instant::now();
    let index = match load_index(index_file) {
        Some(mut idx) => {
            println!("Loaded index from file.");
            let files = dashmap::DashMap::with_capacity(idx.sorted_entries.len());
            idx.sorted_entries.par_iter().for_each(|entry| {
                files.insert(entry.lower_name.clone(), entry.clone());
            });
            idx.files = files;
            idx
        }
        None => {
            println!("No existing index found. Building from scratch...");
            let idx = build_index(root_dir);
            save_index(&idx, index_file);
            idx
        }
    };
    println!("Load/Build index took {:?}", load_start.elapsed());

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    std::fs::create_dir_all(&config.model_dir)?;

    let cfg = SentenceEmbeddingsConfig {
        modules_config_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/modules.json", config.model_dir)))),
        transformer_config_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/config.json", config.model_dir)))),
        transformer_weights_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/pytorch_model.bin", config.model_dir)))),
        dense_config_resource: Some(Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/dense_config.json", config.model_dir))))),
        dense_weights_resource: Some(Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/dense_model.bin", config.model_dir))))),
        sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/sentence_bert_config.json", config.model_dir)))),
        tokenizer_config_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/tokenizer_config.json", config.model_dir)))),
        tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/vocab.txt", config.model_dir)))),
        transformer_type: ModelType::Bert,
        pooling_config_resource: Box::new(RemoteResource::from_pretrained((config.model_id.as_str(), &format!("{}/pooling_config.json", config.model_dir)))),
        device,
        kind: None,
        tokenizer_merges_resource: None,
    };

    let model_start = Instant::now();
    let model = SentenceEmbeddingsModel::new(cfg)?;
    println!("Model loading took {:?}", model_start.elapsed());

    let query_encode_start = Instant::now();
    let query_embedding = model.encode(&[query.to_string()])?[0].clone();
    println!("Query encoding took {:?}", query_encode_start.elapsed());

    println!("Processing documents...");
    // Define ignore patterns for AI scan even after the index is built.
    let ignore_patterns = ["node_modules", "System Volume Information", ".git", "target", "vendor"];
    let docs_start = Instant::now();
    let documents: Vec<(String, String)> = index.sorted_entries
        .par_iter()
        .filter(|entry| {
            let ignore = ignore_patterns.iter().any(|pat| entry.path.contains(pat));
            !ignore &&
            std::fs::metadata(&entry.path).map(|m| m.len() <= config.max_file_size).unwrap_or(false) &&
            document_parser::is_supported_document(&entry.path)
        })
        .filter_map(|entry| {
            document_parser::extract_text(&entry.path)
                .map(|DocumentContent { text, .. }| (entry.path.clone(), text))
        })
        .collect();
    println!("Document processing took {:?}", docs_start.elapsed());
    println!("Found {} processable documents", documents.len());

    println!("Processing documents in batches of {}", config.semantic_batch_size);
    let pb = ProgressBar::new(documents.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} (ETA: {eta})")
        .unwrap()
        .progress_chars("##-"));

    let all_results: Arc<Mutex<Vec<(String, f32)>>> = Arc::new(Mutex::new(Vec::new()));

    let batch_start = Instant::now();
    documents.chunks(config.semantic_batch_size).for_each(|chunk| {

        let texts: Vec<&String> = chunk.iter().map(|(_, content)| content).collect();
        let embeddings = model.encode(&texts).unwrap();
        let batch_results: Vec<(String, f32)> = chunk
            .iter()
            .zip(embeddings.iter())
            .map(|((file_path, _), emb)| {
                let similarity = cosine_similarity(&query_embedding, emb);
                (file_path.clone(), similarity)
            })
            .collect();

        let mut global_results = all_results.lock().unwrap();
        global_results.extend(batch_results);
        global_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        global_results.truncate(config.top_n_results);
        pb.inc(chunk.len() as u64);
    });
    pb.finish_with_message("Done processing files.");
    println!("\nBatch processing took {:?}", batch_start.elapsed());

    let sort_start = Instant::now();
    let mut final_results = all_results.lock().unwrap().clone();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Sorting results took {:?}", sort_start.elapsed());

    println!("\nFinal top results:");
    for (file, score) in final_results.iter().take(config.top_n_results) {
        println!("{} (score: {:.4})", file, score);
    }

    Ok(())
}
