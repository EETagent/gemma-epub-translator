use crate::translate::{translate_texts_with_srt_prompt_with_cancel, LlamaState};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SrtProcessError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid SRT format: {0}")]
    InvalidFormat(String),
    #[error(transparent)]
    Translate(#[from] crate::translate::TranslateError),
    #[error("subtitle index/timing row is malformed")]
    MalformedCue,
    #[error("translation cancelled")]
    Cancelled,
}

#[derive(Clone, Debug)]
struct SrtCue {
    index: String,
    timing: String,
    text_lines: Vec<String>,
}

#[derive(Clone, Debug)]
struct CueContext {
    cue_index: usize,
    text: String,
    is_short: bool,
}

pub fn translate_srt_with_cancel<P, F>(
    state: &Arc<Mutex<LlamaState>>,
    input_path: P,
    source_locale: &str,
    target_locale: &str,
    mut on_progress: F,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<PathBuf, SrtProcessError>
where
    P: AsRef<Path>,
    F: FnMut(usize, usize),
{
    let input_path = input_path.as_ref();
    let content = fs::read_to_string(input_path)?;
    let mut cues = parse_srt(&content)?;

    let contexts = build_cue_contexts(&cues);
    let total = contexts.len();
    if total == 0 {
        let output = output_path_with_locale(input_path, target_locale);
        fs::write(&output, content)?;
        return Ok(output);
    }

    let mut translated_cues: Vec<Option<Vec<String>>> = vec![None; total];
    let mut done = 0usize;
    let batch_size = 24;
    let mut start = 0usize;

    while start < total {
        if is_cancelled(cancel_flag.as_ref()) {
            return Err(SrtProcessError::Cancelled);
        }

        let end = usize::min(start + batch_size, total);
        let batch = &contexts[start..end];
        let prompt_texts: Vec<String> = batch
            .iter()
            .enumerate()
            .map(|(offset, ctx)| build_cue_payload(ctx, &contexts, start + offset))
            .collect();

        let translated_batch = translate_texts_with_srt_prompt_with_cancel(
            state,
            &prompt_texts,
            source_locale,
            target_locale,
            cancel_flag.as_ref(),
        )?;

        if is_cancelled(cancel_flag.as_ref()) {
            return Err(SrtProcessError::Cancelled);
        }

        for (offset, cue_ctx) in batch.iter().enumerate() {
            let raw = translated_batch
                .get(offset)
                .cloned()
                .unwrap_or_else(|| cue_ctx.text.clone());

            let parsed = extract_current_translation(&raw, cue_ctx).and_then(|value| {
                split_translated_lines(&value, cue_line_count(&cues[cue_ctx.cue_index]))
            });

            let source_lines = cues[cue_ctx.cue_index].text_lines.clone();
            let translated_lines = parsed
                .map(|lines| apply_source_formatting(&source_lines, lines))
                .unwrap_or(source_lines);
            translated_cues[start + offset] = Some(translated_lines);
            done += 1;
            on_progress(done, total);
        }

        start = end;
    }

    apply_translations(&mut cues, &contexts, &translated_cues);
    let output_content = serialize_srt(&cues);
    let output = output_path_with_locale(input_path, target_locale);
    fs::write(&output, output_content)?;
    Ok(output)
}

fn parse_srt(content: &str) -> Result<Vec<SrtCue>, SrtProcessError> {
    let normalized = content.replace("\r\n", "\n");
    let blocks = split_srt_blocks(&normalized);
    let mut cues = Vec::new();

    for block in &blocks {
        let mut lines = block
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .into_iter();

        let Some(index) = lines.next() else {
            continue;
        };
        let Some(timing) = lines.next() else {
            return Err(SrtProcessError::InvalidFormat(
                "Missing timing row in SRT block".to_string(),
            ));
        };

        if index.trim().is_empty() || !timing.contains("-->") {
            return Err(SrtProcessError::MalformedCue);
        }

        let text_lines: Vec<String> = lines.map(str::to_string).collect();

        cues.push(SrtCue {
            index: index.to_string(),
            timing: timing.to_string(),
            text_lines,
        });
    }

    if cues.is_empty() {
        return Err(SrtProcessError::InvalidFormat(
            "No subtitle cues found".to_string(),
        ));
    }

    Ok(cues)
}

fn serialize_srt(cues: &[SrtCue]) -> String {
    let mut out = String::new();
    for cue in cues {
        out.push_str(&cue.index);
        out.push('\n');
        out.push_str(&cue.timing);
        out.push('\n');
        for line in &cue.text_lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

fn build_cue_contexts(cues: &[SrtCue]) -> Vec<CueContext> {
    cues.iter()
        .enumerate()
        .filter_map(|(cue_index, cue)| {
            let text = cue
                .text_lines
                .iter()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<&str>>()
                .join("\n");
            if text.is_empty() {
                return None;
            }

            let words = text.split_whitespace().count();
            Some(CueContext {
                cue_index,
                text,
                is_short: words <= 4,
            })
        })
        .collect()
}

fn build_cue_payload(unit: &CueContext, units: &[CueContext], unit_pos: usize) -> String {
    let prev = if unit_pos > 0 {
        units[unit_pos - 1].text.as_str()
    } else {
        ""
    };
    let next = if unit_pos + 1 < units.len() {
        units[unit_pos + 1].text.as_str()
    } else {
        ""
    };

    if unit.is_short {
        format!(
            "<context_prev>\n{prev}\n</context_prev>\n<context_next>\n{next}\n</context_next>\n<current>\n{}\n</current>",
            unit.text
        )
    } else {
        format!("<current>\n{}\n</current>", unit.text)
    }
}

fn extract_current_translation(raw: &str, cue_ctx: &CueContext) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower = trimmed.to_ascii_lowercase();
    if let (Some(start), Some(end)) = (lower.find("<translated>"), lower.find("</translated>")) {
        if end > start {
            let value_start = start + "<translated>".len();
            let value = trimmed[value_start..end].trim();
            if !value.is_empty() {
                if let Some(code_block) = extract_first_code_fence(value) {
                    return Some(code_block);
                }
                return sanitize_model_output(value, cue_ctx);
            }
        }
    }

    if let Some(code_block) = extract_first_code_fence(trimmed) {
        return Some(code_block);
    }

    sanitize_model_output(trimmed, cue_ctx)
}

fn extract_first_code_fence(raw: &str) -> Option<String> {
    let mut in_fence = false;
    let mut lines = Vec::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_fence {
                let content = lines.join("\n").trim().to_string();
                if !content.is_empty() {
                    return Some(content);
                }
                in_fence = false;
                lines.clear();
                continue;
            }

            in_fence = true;
            continue;
        }

        if in_fence {
            lines.push(line.trim_end().to_string());
        }
    }

    None
}

fn sanitize_model_output(raw: &str, cue_ctx: &CueContext) -> Option<String> {
    let mut lines = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("<context_")
            || lower.starts_with("</context_")
            || lower == "<current>"
            || lower == "</current>"
            || lower.starts_with("translate this")
            || lower.starts_with("translate subtitle cues")
            || lower.starts_with("translate only the text inside")
            || lower.starts_with("return only")
            || lower.starts_with("do not output any text outside")
            || lower.starts_with("you are translating")
            || trimmed.starts_with("```")
        {
            continue;
        }
        if trimmed.starts_with('<') && trimmed.ends_with('>') {
            continue;
        }
        lines.push(trimmed.to_string());
    }

    if lines.is_empty() {
        return Some(cue_ctx.text.clone());
    }

    Some(lines.join("\n"))
}

fn apply_source_formatting(source_lines: &[String], translated_lines: Vec<String>) -> Vec<String> {
    if source_lines.is_empty() {
        return translated_lines;
    }

    let max_lines = source_lines.len().max(1);
    let mut aligned = if translated_lines.len() <= max_lines {
        translated_lines
    } else {
        vec![translated_lines.join(" ")]
    };

    if aligned.is_empty() {
        return source_lines.to_vec();
    }

    for (idx, line) in aligned.iter_mut().enumerate() {
        let source = source_lines
            .get(idx)
            .map(String::as_str)
            .unwrap_or_else(|| source_lines.last().map(String::as_str).unwrap_or(""));
        *line = preserve_speaker_prefix(source, line);
        *line = preserve_simple_inline_wrappers(source, line);
    }

    aligned
}

fn preserve_speaker_prefix(source: &str, translated: &str) -> String {
    let source_trimmed = source.trim_start();
    let translated_trimmed = translated.trim_start();
    if source_trimmed.starts_with("- ") && !translated_trimmed.starts_with('-') {
        return format!("- {}", translated_trimmed);
    }
    if source_trimmed.starts_with("– ") && !translated_trimmed.starts_with('–') {
        return format!("– {}", translated_trimmed);
    }
    translated.to_string()
}

fn preserve_simple_inline_wrappers(source: &str, translated: &str) -> String {
    let source_trimmed = source.trim();
    let translated_trimmed = translated.trim();
    for tag in ["i", "b", "u"] {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        if source_trimmed.starts_with(&open)
            && source_trimmed.ends_with(&close)
            && !(translated_trimmed.starts_with(&open) && translated_trimmed.ends_with(&close))
        {
            return format!("{open}{translated_trimmed}{close}");
        }
    }

    translated.to_string()
}

fn apply_translations(
    cues: &mut [SrtCue],
    units: &[CueContext],
    translated_lines: &[Option<Vec<String>>],
) {
    for (idx, unit) in units.iter().enumerate() {
        let Some(translated) = translated_lines.get(idx).and_then(|v| v.as_ref()) else {
            continue;
        };

        if let Some(cue) = cues.get_mut(unit.cue_index) {
            cue.text_lines = translated.clone();
        }
    }
}

fn cue_line_count(cue: &SrtCue) -> usize {
    cue.text_lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .count()
        .max(1)
}

fn split_translated_lines(text: &str, max_lines: usize) -> Option<Vec<String>> {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<String>>();

    if lines.is_empty() {
        return None;
    }

    if lines.len() <= max_lines {
        return Some(lines);
    }

    let merged = lines.join(" ");
    Some(vec![merged])
}

fn split_srt_blocks(normalized: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();

    for line in normalized.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(current.join("\n"));
                current.clear();
            }
            continue;
        }
        current.push(line.to_string());
    }

    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }

    blocks
}

fn is_cancelled(flag: Option<&Arc<AtomicBool>>) -> bool {
    flag.map(|f| !f.load(Ordering::SeqCst)).unwrap_or(false)
}

fn output_path_with_locale(input_path: &Path, target_locale: &str) -> PathBuf {
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("subtitles");
    parent.join(format!("{stem}-{target_locale}.srt"))
}
