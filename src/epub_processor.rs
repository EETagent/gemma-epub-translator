use crate::epub_rebuild::{output_path_with_locale, rebuild_epub_with, RebuildError};
use crate::translate::translate_texts;
use epub::doc::{DocError, EpubDoc};
use lol_html::html_content::TextType;
use lol_html::{text, HtmlRewriter, Settings};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub enum ProcessError {
    Doc(DocError),
    Rebuild(RebuildError),
}

impl From<DocError> for ProcessError {
    fn from(err: DocError) -> Self {
        Self::Doc(err)
    }
}

impl From<RebuildError> for ProcessError {
    fn from(err: RebuildError) -> Self {
        Self::Rebuild(err)
    }
}

#[derive(Debug, Clone)]
pub struct TextSegment {
    pub id: usize,
    pub text: String,
    pub file_name: String,
}

pub struct ExtractionResult {
    pub segments: Vec<TextSegment>,
    pub total_words: usize,
}

pub fn extract_text_segments<P: AsRef<Path>>(path: P) -> Result<ExtractionResult, ProcessError> {
    let mut doc = EpubDoc::new(path)?;
    let mut segments = Vec::new();
    let mut segment_id = 0;
    let mut total_words = 0;

    for spine_item in doc.spine.clone() {
        let Some((html, mime)) = doc.get_resource_str(&spine_item.idref) else {
            continue;
        };
        if !mime.contains("html") && !mime.contains("xhtml") {
            continue;
        }

        let file_name = spine_item.idref.clone();
        let texts = extract_text_from_html(&html);

        for text in texts {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }

            let words = count_words(text);
            total_words += words;

            segments.push(TextSegment {
                id: segment_id,
                text: text.to_string(),
                file_name: file_name.clone(),
            });
            segment_id += 1;
        }
    }

    Ok(ExtractionResult {
        segments,
        total_words,
    })
}

pub fn translate_epub_with_progress<P, F>(
    input_path: P,
    source_locale: &str,
    target_locale: &str,
    mut on_progress: F,
) -> Result<PathBuf, ProcessError>
where
    P: AsRef<Path>,
    F: FnMut(usize, usize),
{
    let input_path = input_path.as_ref();
    let output_path = output_path_with_locale(input_path, target_locale);

    let extraction = extract_text_segments(input_path)?;
    let total = extraction.segments.len();

    if total == 0 {
        rebuild_epub_with(input_path, &output_path, |_, _| None)?;
        return Ok(output_path);
    }

    let mut translation_map: HashMap<String, String> = HashMap::new();

    let batch_size = 32;
    let mut start = 0usize;

    while start < total {
        let end = usize::min(start + batch_size, total);
        let slice = &extraction.segments[start..end];
        let texts: Vec<String> = slice.iter().map(|segment| segment.text.clone()).collect();

        let translated_batch = translate_texts(&texts, source_locale, target_locale);

        for (offset, segment) in slice.iter().enumerate() {
            let translated = translated_batch
                .get(offset)
                .cloned()
                .unwrap_or_else(|| segment.text.clone());
            let index = start + offset;
            println!(
                "Translated {}/{}: {} -> {}",
                index + 1,
                total,
                segment.text,
                translated
            );
            translation_map.insert(segment.text.clone(), translated);
            on_progress(index + 1, total);
        }

        start = end;
    }

    let translation_map = Arc::new(translation_map);

    rebuild_epub_with(input_path, &output_path, |name, bytes| {
        if is_html_name(name) {
            Some(rewrite_html_with_translations(bytes, &translation_map))
        } else {
            None
        }
    })?;

    Ok(output_path)
}

fn extract_text_from_html(html: &str) -> Vec<String> {
    let texts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let texts_writer = texts.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![text!("p,li,h1,h2,h3,h4,h5,h6", move |chunk| {
                if chunk.text_type() == TextType::Data {
                    let text = chunk.as_str().trim();
                    if text.is_empty() {
                        return Ok(());
                    }

                    let normalized = text.replace("&nbsp;", " ");
                    if normalized.trim().is_empty() {
                        return Ok(());
                    }

                    texts_writer.borrow_mut().push(text.to_string());
                }
                Ok(())
            })],
            ..Settings::default()
        },
        |_: &[u8]| {},
    );

    let _ = rewriter.write(html.as_bytes());
    let _ = rewriter.end();

    texts.take()
}

fn rewrite_html_with_translations(bytes: &[u8], translations: &HashMap<String, String>) -> Vec<u8> {
    let html = String::from_utf8_lossy(bytes);
    let mut output = Vec::new();

    let translations = translations.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![text!("p,li,h1,h2,h3,h4,h5,h6", move |chunk| {
                if chunk.text_type() == TextType::Data {
                    let original = chunk.as_str();
                    let trimmed = original.trim();

                    if !trimmed.is_empty() {
                        if let Some(translated) = translations.get(trimmed) {
                            let leading_ws: String =
                                original.chars().take_while(|c| c.is_whitespace()).collect();
                            let trailing_ws: String = original
                                .chars()
                                .rev()
                                .take_while(|c| c.is_whitespace())
                                .collect();
                            let result = format!("{}{}{}", leading_ws, translated, trailing_ws);
                            chunk.set_str(result);
                        }
                    }
                }
                Ok(())
            })],
            ..Settings::default()
        },
        |c: &[u8]| output.extend_from_slice(c),
    );

    let _ = rewriter.write(html.as_bytes());
    let _ = rewriter.end();

    output
}

fn is_html_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".html") || lower.ends_with(".xhtml") || lower.ends_with(".htm")
}

fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}
