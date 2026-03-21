use crate::translate::{translate_texts_with_cancel, LlamaState};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum SrtProcessError {
    Io(std::io::Error),
    InvalidFormat(String),
    Translate(String),
    Cancelled,
}

impl From<std::io::Error> for SrtProcessError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

#[derive(Clone, Debug)]
struct SrtCue {
    index: String,
    timing: String,
    text_lines: Vec<String>,
}

#[derive(Clone, Debug)]
struct CueTextUnit {
    cue_index: usize,
    line_index: usize,
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

    let units = build_text_units(&cues);
    let total = units.len();
    if total == 0 {
        let output = output_path_with_locale(input_path, target_locale);
        fs::write(&output, content)?;
        return Ok(output);
    }

    let mut translated_lines: Vec<Option<String>> = vec![None; total];
    let mut done = 0usize;
    let batch_size = 32;
    let mut start = 0usize;

    while start < total {
        if is_cancelled(cancel_flag.as_ref()) {
            return Err(SrtProcessError::Cancelled);
        }

        let end = usize::min(start + batch_size, total);
        let batch = &units[start..end];
        let prompt_texts: Vec<String> = batch
            .iter()
            .map(|unit| build_contextual_prompt(unit, &units))
            .collect();

        let translated_batch = translate_texts_with_cancel(
            state,
            &prompt_texts,
            source_locale,
            target_locale,
            cancel_flag.as_ref(),
        )
        .map_err(|err| SrtProcessError::Translate(err.to_string()))?;

        if is_cancelled(cancel_flag.as_ref()) {
            return Err(SrtProcessError::Cancelled);
        }

        for (offset, unit) in batch.iter().enumerate() {
            let raw = translated_batch
                .get(offset)
                .cloned()
                .unwrap_or_else(|| unit.text.clone());
            let normalized = extract_current_translation(&raw, &unit.text);
            translated_lines[start + offset] = Some(normalized);
            done += 1;
            on_progress(done, total);
        }

        start = end;
    }

    apply_translations(&mut cues, &units, &translated_lines);
    let output_content = serialize_srt(&cues);
    let output = output_path_with_locale(input_path, target_locale);
    fs::write(&output, output_content)?;
    Ok(output)
}

fn parse_srt(content: &str) -> Result<Vec<SrtCue>, SrtProcessError> {
    let normalized = content.replace("\r\n", "\n");
    let blocks: Vec<&str> = normalized.split("\n\n").collect();
    let mut cues = Vec::new();

    for block in blocks {
        let mut lines = block
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty());

        let Some(index) = lines.next() else {
            continue;
        };
        let Some(timing) = lines.next() else {
            return Err(SrtProcessError::InvalidFormat(
                "Missing timing row in SRT block".to_string(),
            ));
        };
        let text_lines: Vec<String> = lines.map(|line| line.to_string()).collect();

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

fn build_text_units(cues: &[SrtCue]) -> Vec<CueTextUnit> {
    let mut units = Vec::new();
    for (cue_index, cue) in cues.iter().enumerate() {
        for (line_index, line) in cue.text_lines.iter().enumerate() {
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            let words = text.split_whitespace().count();
            units.push(CueTextUnit {
                cue_index,
                line_index,
                text: text.to_string(),
                is_short: words <= 3,
            });
        }
    }
    units
}

fn build_contextual_prompt(unit: &CueTextUnit, units: &[CueTextUnit]) -> String {
    let unit_pos = units
        .iter()
        .position(|u| u.cue_index == unit.cue_index && u.line_index == unit.line_index)
        .unwrap_or(0);

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
            "Context previous line: {prev}\nCurrent line: {}\nContext next line: {next}\nTranslate only the Current line.",
            unit.text
        )
    } else {
        unit.text.clone()
    }
}

fn extract_current_translation(raw: &str, fallback: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return fallback.to_string();
    }

    if let Some(line) = trimmed
        .lines()
        .find(|line| line.to_ascii_lowercase().starts_with("current line:"))
    {
        let translated = line
            .split_once(':')
            .map(|(_, value)| value.trim())
            .unwrap_or_default();
        if !translated.is_empty() {
            return translated.to_string();
        }
    }

    trimmed
        .lines()
        .next()
        .unwrap_or(fallback)
        .trim()
        .to_string()
}

fn apply_translations(
    cues: &mut [SrtCue],
    units: &[CueTextUnit],
    translated_lines: &[Option<String>],
) {
    for (idx, unit) in units.iter().enumerate() {
        let Some(translated) = translated_lines.get(idx).and_then(|v| v.as_ref()) else {
            continue;
        };

        if let Some(cue) = cues.get_mut(unit.cue_index) {
            if let Some(line) = cue.text_lines.get_mut(unit.line_index) {
                *line = translated.clone();
            }
        }
    }
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
