use crate::processors::epub_rebuild::{output_path_with_locale, rebuild_epub_with, RebuildError};
use crate::translate::{translate_texts_with_cancel, LlamaState, TranslateError};
use epub::doc::{DocError, EpubDoc};
use lol_html::html_content::{ContentType, TextType};
use lol_html::{element, text, HtmlRewriter, Settings};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum ProcessError {
    Doc(DocError),
    Rebuild(RebuildError),
    Translate(TranslateError),
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

impl From<TranslateError> for ProcessError {
    fn from(err: TranslateError) -> Self {
        Self::Translate(err)
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
    state: &Arc<Mutex<LlamaState>>,
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

        let translated_batch = translate_texts_with_cancel(
            state,
            &texts,
            source_locale,
            target_locale,
            cancel_flag.as_ref(),
        )?;

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

const BLOCK_TEXT_CONTAINERS: &str = "p,li,h1,h2,h3,h4,h5,h6,div";
const INLINE_TEXT_CONTAINERS: &str = "span";
const TRANSLATABLE_TEXT_CONTAINERS: &str = "p,li,h1,h2,h3,h4,h5,h6,div,span";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureKind {
    Block,
    Span,
}

#[derive(Debug)]
struct CaptureFrame {
    kind: CaptureKind,
    text: String,
    has_block_descendant: bool,
}

fn push_capture_frame(
    stack: &std::rc::Rc<std::cell::RefCell<Vec<CaptureFrame>>>,
    kind: CaptureKind,
) {
    let mut frames = stack.borrow_mut();
    if kind == CaptureKind::Block {
        if let Some(parent) = frames.last_mut() {
            if parent.kind == CaptureKind::Block {
                parent.has_block_descendant = true;
            }
        }
    }

    frames.push(CaptureFrame {
        kind,
        text: String::new(),
        has_block_descendant: false,
    });
}

fn pop_capture_frame(
    stack: &std::rc::Rc<std::cell::RefCell<Vec<CaptureFrame>>>,
) -> Option<(CaptureFrame, bool)> {
    let frame = stack.borrow_mut().pop()?;
    let has_block_ancestor = stack
        .borrow()
        .iter()
        .any(|ancestor| ancestor.kind == CaptureKind::Block);
    Some((frame, has_block_ancestor))
}

fn should_emit_frame(frame: &CaptureFrame, has_block_ancestor: bool) -> bool {
    match frame.kind {
        CaptureKind::Block => !frame.has_block_descendant,
        CaptureKind::Span => !has_block_ancestor,
    }
}

fn normalized_segment(content: &str) -> Option<String> {
    let normalized = content.replace("&nbsp;", " ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn translated_segment(content: &str, translations: &HashMap<String, String>) -> Option<String> {
    let normalized = content.replace("&nbsp;", " ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }

    let leading_ws: String = normalized
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();
    let trailing_ws: String = normalized
        .chars()
        .rev()
        .take_while(|c| c.is_whitespace())
        .collect::<String>()
        .chars()
        .rev()
        .collect();

    let translated = translations
        .get(trimmed)
        .cloned()
        .unwrap_or_else(|| trimmed.to_string());
    Some(format!("{}{}{}", leading_ws, translated, trailing_ws))
}

fn extract_text_from_html(html: &str) -> Vec<String> {
    let texts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let stack = std::rc::Rc::new(std::cell::RefCell::new(Vec::<CaptureFrame>::new()));
    let stack_for_blocks = stack.clone();
    let stack_for_blocks_end = stack.clone();
    let texts_for_blocks_end = texts.clone();
    let stack_for_spans = stack.clone();
    let stack_for_spans_end = stack.clone();
    let texts_for_spans_end = texts.clone();
    let stack_for_text = stack.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![
                element!(BLOCK_TEXT_CONTAINERS, move |el| {
                    push_capture_frame(&stack_for_blocks, CaptureKind::Block);
                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_blocks_end.clone();
                        let texts_for_end = texts_for_blocks_end.clone();
                        handlers.push(Box::new(move |_end| {
                            if let Some((frame, has_block_ancestor)) =
                                pop_capture_frame(&stack_for_end)
                            {
                                if should_emit_frame(&frame, has_block_ancestor) {
                                    if let Some(normalized) = normalized_segment(&frame.text) {
                                        texts_for_end.borrow_mut().push(normalized);
                                    }
                                }
                            }
                            Ok(())
                        }));
                    } else {
                        let _ = pop_capture_frame(&stack_for_blocks);
                    }
                    Ok(())
                }),
                element!(INLINE_TEXT_CONTAINERS, move |el| {
                    let should_capture = {
                        let frames = stack_for_spans.borrow();
                        let has_block_ancestor =
                            frames.iter().any(|frame| frame.kind == CaptureKind::Block);
                        let has_span_ancestor =
                            frames.iter().any(|frame| frame.kind == CaptureKind::Span);
                        !has_block_ancestor && !has_span_ancestor
                    };

                    if should_capture {
                        push_capture_frame(&stack_for_spans, CaptureKind::Span);
                    }

                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_spans_end.clone();
                        let texts_for_end = texts_for_spans_end.clone();
                        handlers.push(Box::new(move |_end| {
                            if should_capture {
                                if let Some((frame, has_block_ancestor)) =
                                    pop_capture_frame(&stack_for_end)
                                {
                                    if should_emit_frame(&frame, has_block_ancestor) {
                                        if let Some(normalized) = normalized_segment(&frame.text) {
                                            texts_for_end.borrow_mut().push(normalized);
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }));
                    } else if should_capture {
                        let _ = pop_capture_frame(&stack_for_spans);
                    }

                    Ok(())
                }),
                text!(TRANSLATABLE_TEXT_CONTAINERS, move |chunk| {
                    if chunk.text_type() == TextType::Data {
                        let text = chunk.as_str();
                        if !text.is_empty() {
                            let mut frames = stack_for_text.borrow_mut();
                            for frame in frames.iter_mut() {
                                frame.text.push_str(text);
                            }
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

    let stack = std::rc::Rc::new(std::cell::RefCell::new(Vec::<CaptureFrame>::new()));
    let stack_for_blocks = stack.clone();
    let stack_for_blocks_end = stack.clone();
    let translations_for_blocks_end = translations.clone();
    let stack_for_spans = stack.clone();
    let stack_for_spans_end = stack.clone();
    let translations_for_spans_end = translations.clone();
    let stack_for_text = stack.clone();

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![
                element!(BLOCK_TEXT_CONTAINERS, move |el| {
                    push_capture_frame(&stack_for_blocks, CaptureKind::Block);
                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_blocks_end.clone();
                        let translations_for_end = translations_for_blocks_end.clone();
                        handlers.push(Box::new(move |end| {
                            if let Some((frame, has_block_ancestor)) =
                                pop_capture_frame(&stack_for_end)
                            {
                                if should_emit_frame(&frame, has_block_ancestor) {
                                    if let Some(result) = translated_segment(
                                        &frame.text,
                                        translations_for_end.as_ref(),
                                    ) {
                                        end.before(&result, ContentType::Text);
                                    }
                                }
                            }
                            Ok(())
                        }));
                    } else {
                        let _ = pop_capture_frame(&stack_for_blocks);
                    }
                    Ok(())
                }),
                element!(INLINE_TEXT_CONTAINERS, move |el| {
                    let should_capture = {
                        let frames = stack_for_spans.borrow();
                        let has_block_ancestor =
                            frames.iter().any(|frame| frame.kind == CaptureKind::Block);
                        let has_span_ancestor =
                            frames.iter().any(|frame| frame.kind == CaptureKind::Span);
                        !has_block_ancestor && !has_span_ancestor
                    };

                    if should_capture {
                        push_capture_frame(&stack_for_spans, CaptureKind::Span);
                    }

                    if let Some(handlers) = el.end_tag_handlers() {
                        let stack_for_end = stack_for_spans_end.clone();
                        let translations_for_end = translations_for_spans_end.clone();
                        handlers.push(Box::new(move |end| {
                            if should_capture {
                                if let Some((frame, has_block_ancestor)) =
                                    pop_capture_frame(&stack_for_end)
                                {
                                    if should_emit_frame(&frame, has_block_ancestor) {
                                        if let Some(result) = translated_segment(
                                            &frame.text,
                                            translations_for_end.as_ref(),
                                        ) {
                                            end.before(&result, ContentType::Text);
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }));
                    } else if should_capture {
                        let _ = pop_capture_frame(&stack_for_spans);
                    }

                    Ok(())
                }),
                text!(TRANSLATABLE_TEXT_CONTAINERS, move |chunk| {
                    if chunk.text_type() == TextType::Data {
                        let text = chunk.as_str();
                        let mut frames = stack_for_text.borrow_mut();
                        if !frames.is_empty() {
                            for frame in frames.iter_mut() {
                                frame.text.push_str(text);
                            }
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
