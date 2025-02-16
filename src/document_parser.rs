use calamine::{open_workbook, Reader, Xlsx};
use lopdf::Document;
use pulldown_cmark::{Parser, html::push_html};
use std::fs::File;
use std::io::Read;

pub struct DocumentContent {
    pub text: String,
    pub metadata: String,
}

pub fn is_supported_document(path: &str) -> bool {
    let lower_path = path.to_lowercase();
    let supported_extensions = [
        ".txt", ".doc", ".docx", ".xls", ".xlsx",
        ".pdf", ".md", ".json", "csv", ".xml",
    ];
    supported_extensions.iter().any(|&ext| lower_path.ends_with(ext))
}

pub fn extract_text(path: &str) -> Option<DocumentContent> {
    let path_lower = path.to_lowercase();
    let content = match () {
        _ if path_lower.ends_with(".txt") || 
          path_lower.ends_with(".json") ||
          path_lower.ends_with(".csv") ||
          path_lower.ends_with(".xml") => read_text_file(path),
        _ if path_lower.ends_with(".pdf") => extract_pdf_text(path),
        _ if path_lower.ends_with(".xlsx") || path_lower.ends_with(".xls") => extract_excel_text(path),
        _ if path_lower.ends_with(".docx") => extract_docx_text(path),
        _ if path_lower.ends_with(".md") => extract_markdown_text(path),
        _ => read_text_file(path), // Try as text for other formats
    }?;

    Some(DocumentContent {
        text: content,
        metadata: format!("Path: {}", path),
    })
}

fn read_text_file(path: &str) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut content = String::new();
    file.read_to_string(&mut content).ok()?;
    Some(content)
}

fn extract_pdf_text(path: &str) -> Option<String> {
    if let Ok(doc) = Document::load(path) {
        let mut text = String::new();
        for page_number in 1..=doc.get_pages().len() {
            if let Ok(page_text) = doc.extract_text(&[page_number.try_into().unwrap()]) {
                text.push_str(&page_text);
                text.push('\n');
            }
        }
        Some(text)
    } else {
        None
    }
}

fn extract_excel_text(path: &str) -> Option<String> {
    let mut workbook: Xlsx<_> = open_workbook(path).ok()?;
    let mut text = String::new();
    
    // Clone sheet names to avoid borrow issues
    let sheet_names = workbook.sheet_names().to_owned();
    
    for sheet_name in sheet_names {
        // Change pattern matching from Option to directly matching the Result.
        if let Ok(range) = workbook.worksheet_range(&sheet_name) {
            for row in range.rows() {
                for cell in row {
                    text.push_str(&cell.to_string());
                    text.push('\t');
                }
                text.push('\n');
            }
        }
    }
    
    Some(text)
}

fn extract_docx_text(path: &str) -> Option<String> {
    // Basic implementation - just read as text for now
    read_text_file(path)
}

fn extract_markdown_text(path: &str) -> Option<String> {
    let content = read_text_file(path)?;
    let mut html = String::new();
    let parser = Parser::new(&content);
    push_html(&mut html, parser);
    Some(html)
}
