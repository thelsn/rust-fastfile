use serde::Deserialize;
use std::env;

#[derive(Deserialize)]
pub struct Config {
    pub root_dir: String,
    pub index_file: String,
    pub max_file_size: u64,
    pub model_id: String,
    pub model_dir: String,
    pub semantic_batch_size: usize,
    pub top_n_results: usize,
    pub update_cooldown_secs: u64,
    pub rayon_thread_count: usize,
    // Add more config options here if needed
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut cfg: Config = toml::from_str(&content)?;
        // Replace "{user}" with the current username
        if cfg.root_dir.contains("{user}") {
            let user = std::env::var("USERNAME").unwrap_or_else(|_| "default".to_string());
            cfg.root_dir = cfg.root_dir.replace("{user}", &user);
        }
        Ok(cfg)
    }
}
