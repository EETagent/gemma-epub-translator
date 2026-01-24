use crate::epub_rebuild::{output_path_with_locale, rebuild_epub_with, RebuildError};
use crate::translate::translate_texts_with_cancel;
use epub::doc::{DocError, EpubDoc};
use lol_html::html_content::{ContentType, TextType};
use lol_html::{element, text, HtmlRewriter, Settings};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub enum ProcessError {
    Doc(DocError),
    Rebuild(RebuildError),
    Cancelled,
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

pub fn translate_epub_with_cancel<P, F>(
    input_path: P,
    source_locale: &str,
    target_locale: &str,
    mut on_progress: F,
    cancel_flag: Option<Arc<AtomicBool>>,
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
        if let Some(ref flag) = cancel_flag {
            if !flag.load(Ordering::SeqCst) {
                return Err(ProcessError::Cancelled);
            }
        }

        let end = usize::min(start + batch_size, total);
        let slice = &extraction.segments[start..end];
        let texts: Vec<String> = slice.iter().map(|segment| segment.text.clone()).collect();

        let translated_batch =
            translate_texts_with_cancel(&texts, source_locale, target_locale, cancel_flag.as_ref());

        if let Some(ref flag) = cancel_flag {
            if !flag.load(Ordering::SeqCst) {
                return Err(ProcessError::Cancelled);
            }
        }

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

    if let Some(ref flag) = cancel_flag {
        if !flag.load(Ordering::SeqCst) {
            return Err(ProcessError::Cancelled);
        }
    }

    let translation_map = Arc::new(translation_map);

    rebuild_epub_with(input_path, &output_path, |name, bytes| {
        if is_html_name(name) {
            Some(rewrite_html_with_translations(
                bytes,
                translation_map.clone(),
            ))
        } else {
            None
        }
    })?;

    Ok(output_path)
}

fn extract_text_from_html(html: &str) -> Vec<String> {
    let texts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let stack = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let stack_for_element = stack.clone();
    let stack_for_text = stack.clone();
    let texts_for_end = texts.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![
                element!("p,li,h1,h2,h3,h4,h5,h6", move |el| {
                    stack_for_element.borrow_mut().push(String::new());
                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_element.clone();
                        let texts_for_end = texts_for_end.clone();
                        handlers.push(Box::new(move |_end| {
                            if let Some(content) = stack_for_end.borrow_mut().pop() {
                                let normalized = content.replace("&nbsp;", " ");
                                let trimmed = normalized.trim();
                                if !trimmed.is_empty() {
                                    texts_for_end.borrow_mut().push(trimmed.to_string());
                                }
                            }
                            Ok(())
                        }));
                    }
                    Ok(())
                }),
                text!("p,li,h1,h2,h3,h4,h5,h6", move |chunk| {
                    if chunk.text_type() == TextType::Data {
                        if let Some(current) = stack_for_text.borrow_mut().last_mut() {
                            current.push_str(chunk.as_str());
                        }
                    }
                    Ok(())
                }),
            ],
            ..Settings::default()
        },
        |_: &[u8]| {},
    );

    let _ = rewriter.write(html.as_bytes());
    let _ = rewriter.end();

    texts.take()
}

fn rewrite_html_with_translations(
    bytes: &[u8],
    translations: Arc<HashMap<String, String>>,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());

    let stack = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let stack_for_element = stack.clone();
    let stack_for_text = stack.clone();
    let translations_for_end = translations.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![
                element!("p,li,h1,h2,h3,h4,h5,h6", move |el| {
                    stack_for_element.borrow_mut().push(String::new());
                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_element.clone();
                        let translations_for_end = translations_for_end.clone();
                        handlers.push(Box::new(move |end| {
                            if let Some(content) = stack_for_end.borrow_mut().pop() {
                                let normalized = content.replace("&nbsp;", " ");
                                let trimmed = normalized.trim();
                                if trimmed.is_empty() {
                                    return Ok(());
                                }

                                let leading_ws: String = normalized
                                    .chars()
                                    .take_while(|c| c.is_whitespace())
                                    .collect();
                                let trailing_ws: String = normalized
                                    .chars()
                                    .rev()
                                    .take_while(|c| c.is_whitespace())
                                    .collect();
                                let translated = translations_for_end
                                    .get(trimmed)
                                    .cloned()
                                    .unwrap_or_else(|| trimmed.to_string());
                                let result = format!("{}{}{}", leading_ws, translated, trailing_ws);
                                end.before(&result, ContentType::Text);
                            }
                            Ok(())
                        }));
                    }
                    Ok(())
                }),
                text!("p,li,h1,h2,h3,h4,h5,h6", move |chunk| {
                    if chunk.text_type() == TextType::Data {
                        if let Some(current) = stack_for_text.borrow_mut().last_mut() {
                            current.push_str(chunk.as_str());
                            chunk.replace("", ContentType::Text);
                        }
                    }
                    Ok(())
                }),
            ],
            ..Settings::default()
        },
        |c: &[u8]| output.extend_from_slice(c),
    );

    let _ = rewriter.write(bytes);
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
