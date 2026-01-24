use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use std::env;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// translategemma-4b-it-q8_0.gguf
const MODEL_FILE: &str = "translategemma-12b-it.Q4_K_M.gguf";
const CTX_SIZE: u32 = 2048 * 4;
const N_BATCH: u32 = 2048 * 4;
const MAX_OUTPUT_TOKENS: usize = 512 * 4;
const SHORT_OUTPUT_TOKENS: usize = 16;
const SHORT_INPUT_TOKENS: usize = 4;
const MAX_SEQ_BATCH: usize = 8 * 2;
const STOP_SEQUENCE: &str = "<end_of_turn>";
const TOP_K: i32 = 64;
const TOP_P: f32 = 0.95;
const TOP_P_MIN_KEEP: usize = 1;

const PROMPT_PREFIX_TEMPLATE: &str = "<bos><start_of_turn>user\nYou are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.\nProduce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:\n\n\n";
const PROMPT_SUFFIX: &str = "<end_of_turn>\n<start_of_turn>model\n";

pub struct LlamaState {
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: Box<LlamaModel>,
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    max_seq_batch: usize,
    stop_tokens: Vec<LlamaToken>,
    cached_prefix: Option<String>,
    cached_prefix_len: usize,
    cached_prompt_suffix_len: usize,
}

unsafe impl Send for LlamaState {}
unsafe impl Sync for LlamaState {}

pub fn create_state() -> LlamaState {
    init_state()
}

pub fn translate_text_with_state(
    state: &Arc<Mutex<LlamaState>>,
    text: &str,
    source_locale: &str,
    target_locale: &str,
) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let results =
        translate_texts_with_state(state, &[text.to_string()], source_locale, target_locale);
    results
        .into_iter()
        .next()
        .unwrap_or_else(|| text.to_string())
}

pub fn translate_texts_with_state(
    state: &Arc<Mutex<LlamaState>>,
    texts: &[String],
    source_locale: &str,
    target_locale: &str,
) -> Vec<String> {
    translate_texts_with_cancel(state, texts, source_locale, target_locale, None)
}

pub struct BatchLimits {
    pub max_prompt_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_sequences: usize,
}

struct TextChunk {
    text: String,
    tokens_len: usize,
}

#[derive(Clone, Copy, Debug)]
struct PromptKey {
    parent_index: usize,
    chunk_index: usize,
    is_short_input: bool,
    is_single_word: bool,
}

pub fn translate_texts_with_cancel(
    state: &Arc<Mutex<LlamaState>>,
    texts: &[String],
    source_locale: &str,
    target_locale: &str,
    cancel_flag: Option<&Arc<AtomicBool>>,
) -> Vec<String> {
    if texts.is_empty() {
        return Vec::new();
    }

    let mut state = state.lock().expect("llama state lock");
    let limits = BatchLimits {
        max_prompt_tokens: CTX_SIZE as usize - MAX_OUTPUT_TOKENS,
        max_batch_tokens: state.ctx.n_batch() as usize,
        max_sequences: state.max_seq_batch.saturating_sub(1),
    };
    let mut outputs = vec![String::new(); texts.len()];
    let mut prompts = Vec::new();
    let mut chunk_results: Vec<Vec<Option<String>>> = vec![Vec::new(); texts.len()];
    let prompt_prefix = build_prompt_prefix(source_locale, target_locale);

    ensure_suffix_cached(&mut state);
    // TODO: Error handling v translate_texts_with_cancel
    ensure_prefix_cached(&mut state, &prompt_prefix).unwrap();

    for (index, text) in texts.iter().enumerate() {
        if text.trim().is_empty() {
            outputs[index] = text.to_string();
            continue;
        }

        let chunks = split_text_to_fit(text, &state, &limits);
        if chunks.is_empty() {
            outputs[index] = text.to_string();
            continue;
        }

        chunk_results[index] = vec![None; chunks.len()];
        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            let is_short_input = is_short_input_text_with_len(chunk.tokens_len);
            let is_single_word = is_single_word_input(&chunk.text);
            prompts.push((
                PromptKey {
                    parent_index: index,
                    chunk_index,
                    is_short_input,
                    is_single_word,
                },
                chunk.text,
            ));
        }
    }

    if prompts.is_empty() {
        return outputs;
    }

    match run_batch_inference(&mut state, &prompts, cancel_flag) {
        Ok(results) => {
            for (key, translated) in results {
                if let Some(chunks) = chunk_results.get_mut(key.parent_index) {
                    if let Some(slot) = chunks.get_mut(key.chunk_index) {
                        *slot = Some(translated);
                    }
                }
            }
        }
        Err(_) => {
            for (key, _) in prompts {
                outputs[key.parent_index] = texts[key.parent_index].to_string();
            }
        }
    }

    for (index, chunks) in chunk_results.into_iter().enumerate() {
        if chunks.is_empty() || !outputs[index].is_empty() {
            continue;
        }

        if chunks.iter().any(|chunk| chunk.is_none()) {
            outputs[index] = texts[index].to_string();
            continue;
        }

        let joined = chunks
            .into_iter()
            .filter_map(|chunk| chunk)
            .collect::<Vec<String>>()
            .join(" ");
        outputs[index] = joined;
    }

    outputs
}

fn run_batch_inference(
    state: &mut LlamaState,
    prompts: &[(PromptKey, String)],
    cancel_flag: Option<&Arc<AtomicBool>>,
) -> Result<Vec<(PromptKey, String)>, llama_cpp_2::LlamaCppError> {
    let max_batch = state.ctx.n_batch() as usize;
    let max_sequences = state.max_seq_batch.saturating_sub(1);
    let mut results = Vec::with_capacity(prompts.len());
    let mut index = 0;
    let prefix_len = state.cached_prefix_len;

    while index < prompts.len() {
        if let Some(flag) = cancel_flag {
            if !flag.load(Ordering::SeqCst) {
                clear_transient_sequences(state);
                return Ok(results);
            }
        }

        let mut batch_indices = Vec::new();
        let mut batch_tokens = Vec::new();
        let mut batch_max_output = MAX_OUTPUT_TOKENS;
        let mut token_count = 0usize;

        while index < prompts.len() {
            let prompt = format!("{}{}", prompts[index].1, PROMPT_SUFFIX);
            let tokens = state
                .model
                .str_to_token(&prompt, AddBos::Never)
                .unwrap_or_default();
            if tokens.is_empty() {
                results.push((prompts[index].0, String::new()));
                index += 1;
                continue;
            }

            let token_len = tokens.len();
            let prompt_max_output = if prompts[index].0.is_short_input {
                SHORT_OUTPUT_TOKENS
            } else {
                MAX_OUTPUT_TOKENS
            };
            if batch_tokens.len() >= max_sequences {
                break;
            }

            if !batch_tokens.is_empty() && token_count + token_len > max_batch {
                break;
            }

            if !batch_tokens.is_empty() && prompt_max_output != batch_max_output {
                break;
            }

            if batch_tokens.is_empty() {
                batch_max_output = prompt_max_output;
            }

            token_count += token_len;
            batch_indices.push(prompts[index].0);
            batch_tokens.push(tokens);
            index += 1;
        }

        if batch_tokens.is_empty() {
            continue;
        }

        let outputs = run_batch_with_tokens(
            state,
            prefix_len,
            &batch_tokens,
            batch_max_output,
            cancel_flag,
        )?;
        for (prompt_index, output) in batch_indices.into_iter().zip(outputs.into_iter()) {
            let filtered = if prompt_index.is_single_word {
                filter_single_word_output(&output)
            } else {
                output
            };
            results.push((prompt_index, filtered));
        }
    }

    Ok(results)
}

fn run_batch_with_tokens(
    state: &mut LlamaState,
    prefix_len: usize,
    prompt_tokens: &[Vec<LlamaToken>],
    max_output_tokens: usize,
    cancel_flag: Option<&Arc<AtomicBool>>,
) -> Result<Vec<String>, llama_cpp_2::LlamaCppError> {
    let stop_len = state.stop_tokens.len();
    clear_transient_sequences(state);

    let total_tokens: usize = prompt_tokens.iter().map(|tokens| tokens.len()).sum();
    let seq_count = prompt_tokens.len();
    let seq_max = (seq_count + 1) as i32;

    for seq_id in 1..=seq_count {
        state
            .ctx
            .copy_kv_cache_seq(0, seq_id as i32, None, None)
            .ok();
    }

    let mut batch = LlamaBatch::new(total_tokens, seq_max);
    let mut logits_indices = Vec::with_capacity(seq_count);
    let mut offset = 0usize;

    for (seq_id, tokens) in prompt_tokens.iter().enumerate() {
        let seq_id = seq_id + 1;
        for (token_index, token) in tokens.iter().enumerate() {
            let pos = prefix_len + token_index;
            let is_last = token_index + 1 == tokens.len();
            batch
                .add(*token, pos as i32, &[seq_id as i32], is_last)
                .ok();
        }
        let last_offset = offset + tokens.len().saturating_sub(1);
        logits_indices.push(last_offset as i32);
        offset += tokens.len();
    }

    state.ctx.decode(&mut batch).ok();

    let mut output_tokens = vec![Vec::with_capacity(max_output_tokens); seq_count];
    let mut positions: Vec<i32> = prompt_tokens
        .iter()
        .map(|tokens| (prefix_len + tokens.len()) as i32)
        .collect();
    let mut active = vec![true; seq_count];
    let mut samplers: Vec<LlamaSampler> = (0..seq_count)
        .map(|seq_index| build_sampler(0x1234_u32.wrapping_add(seq_index as u32)))
        .collect();

    for _ in 0..max_output_tokens {
        if let Some(flag) = cancel_flag {
            if !flag.load(Ordering::SeqCst) {
                clear_transient_sequences(state);
                let outputs = output_tokens
                    .iter()
                    .map(|tokens| {
                        state
                            .model
                            .tokens_to_str(tokens, Special::Plaintext)
                            .unwrap_or_default()
                            .trim()
                            .to_string()
                    })
                    .collect();
                return Ok(outputs);
            }
        }

        let mut next_tokens = Vec::new();
        let mut active_indices = Vec::new();

        for seq_index in 0..seq_count {
            if !active[seq_index] {
                continue;
            }

            let logits = state.ctx.get_logits_ith(logits_indices[seq_index]);
            let mut data_array = LlamaTokenDataArray::from_iter(
                (0..state.model.n_vocab())
                    .zip(logits.iter())
                    .map(|(i, logit)| LlamaTokenData::new(LlamaToken::new(i), *logit, 0.0)),
                false,
            );

            data_array.apply_sampler(&samplers[seq_index]);
            let token = data_array
                .selected_token()
                .unwrap_or_else(|| data_array.sample_token_greedy());
            samplers[seq_index].accept(token);
            if state.model.is_eog_token(token) {
                active[seq_index] = false;
                continue;
            }

            output_tokens[seq_index].push(token);
            if stop_len > 0
                && output_tokens[seq_index].len() >= stop_len
                && tokens_end_with(&output_tokens[seq_index], &state.stop_tokens)
            {
                let new_len = output_tokens[seq_index].len().saturating_sub(stop_len);
                output_tokens[seq_index].truncate(new_len);
                active[seq_index] = false;
                continue;
            }
            next_tokens.push(token);
            active_indices.push(seq_index);
        }

        if active_indices.is_empty() {
            break;
        }

        let mut token_batch = LlamaBatch::new(active_indices.len(), seq_max);
        for (batch_index, seq_index) in active_indices.iter().enumerate() {
            let pos = positions[*seq_index];
            token_batch
                .add(
                    next_tokens[batch_index],
                    pos,
                    &[*seq_index as i32 + 1],
                    true,
                )
                .ok();
            positions[*seq_index] += 1;
            logits_indices[*seq_index] = batch_index as i32;
        }

        state.ctx.decode(&mut token_batch).ok();
    }

    let outputs = output_tokens
        .iter()
        .map(|tokens| {
            state
                .model
                .tokens_to_str(tokens, Special::Plaintext)
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .collect();

    Ok(outputs)
}

fn init_state() -> LlamaState {
    let backend = LlamaBackend::init().expect("llama backend init");
    let model_params = LlamaModelParams::default();

    let model = Box::new(
        LlamaModel::load_from_file(&backend, &model_path(), &model_params).expect("load model"),
    );

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CTX_SIZE))
        .with_n_batch(N_BATCH)
        .with_n_seq_max(MAX_SEQ_BATCH as u32)
        .with_flash_attention_policy(1);

    // TODO: The model is stored in the same struct as the context, so the model will outlive the context. We use unsafe to extend the lifetime.
    let model_ref: &'static LlamaModel = unsafe { &*(&*model as *const LlamaModel) };

    let ctx = model_ref
        .new_context(&backend, ctx_params)
        .expect("init context");
    let stop_tokens = model
        .str_to_token(STOP_SEQUENCE, AddBos::Never)
        .unwrap_or_default();

    LlamaState {
        backend,
        model,
        ctx,
        max_seq_batch: MAX_SEQ_BATCH,
        stop_tokens,
        cached_prefix: None,
        cached_prefix_len: 0,
        cached_prompt_suffix_len: 0,
    }
}

fn ensure_suffix_cached(state: &mut LlamaState) {
    if state.cached_prompt_suffix_len > 0 {
        return;
    }

    let suffix_tokens = state
        .model
        .str_to_token(PROMPT_SUFFIX, AddBos::Never)
        .unwrap_or_default();
    state.cached_prompt_suffix_len = suffix_tokens.len();
}

fn ensure_prefix_cached(
    state: &mut LlamaState,
    prompt_prefix: &str,
) -> Result<usize, llama_cpp_2::LlamaCppError> {
    if state.cached_prefix.as_deref() == Some(prompt_prefix) {
        return Ok(state.cached_prefix_len);
    }
    println!("Updating cached prompt prefix...");

    state.ctx.clear_kv_cache();
    state.cached_prefix = None;
    state.cached_prefix_len = 0;

    let prefix_tokens = state
        .model
        .str_to_token(prompt_prefix, AddBos::Never)
        .unwrap_or_default();
    let prefix_tokens_len = prefix_tokens.len();
    if prefix_tokens_len > 0 {
        let mut prefix_batch = LlamaBatch::new(prefix_tokens_len, 1);
        prefix_batch.add_sequence(&prefix_tokens, 0, false).ok();
        state.ctx.decode(&mut prefix_batch)?;
    }

    state.cached_prefix = Some(prompt_prefix.to_string());
    state.cached_prefix_len = prefix_tokens_len;

    Ok(prefix_tokens_len)
}

fn clear_transient_sequences(state: &mut LlamaState) {
    let max_sequences = state.max_seq_batch.saturating_sub(1);
    for seq_id in 1..=max_sequences {
        state
            .ctx
            .clear_kv_cache_seq(Some(seq_id as u32), None, None)
            .ok();
    }
}

fn build_prompt_prefix(source_locale: &str, target_locale: &str) -> String {
    let (source_lang, source_code) = locale_to_lang(source_locale);
    let (target_lang, target_code) = locale_to_lang(target_locale);

    PROMPT_PREFIX_TEMPLATE
        .replace("{SOURCE_LANG}", source_lang.as_str())
        .replace("{SOURCE_CODE}", source_code.as_str())
        .replace("{TARGET_LANG}", target_lang.as_str())
        .replace("{TARGET_CODE}", target_code.as_str())
}

fn model_path() -> PathBuf {
    let model_name = Path::new(MODEL_FILE);
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            if let Some(contents_dir) = exe_dir.parent() {
                let resources_dir = contents_dir.join("Resources");
                let candidate = resources_dir.join(model_name);
                if candidate.exists() {
                    return candidate;
                }

                let models_candidate = resources_dir.join("models").join(model_name);
                if models_candidate.exists() {
                    return models_candidate;
                }
            }
        }
    }

    Path::new("models").join(model_name)
}

fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

fn text_token_len(state: &LlamaState, text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    state
        .model
        .str_to_token(text, AddBos::Never)
        .unwrap_or_default()
        .len()
}

fn split_text_to_fit(
    text: &str,
    state: &LlamaState,
    limits: &BatchLimits,
) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let max_text_tokens = limits
        .max_prompt_tokens
        .saturating_sub(state.cached_prefix_len)
        .saturating_sub(state.cached_prompt_suffix_len);
    let full_len = text_token_len(state, text);
    if full_len <= max_text_tokens {
        return vec![TextChunk {
            text: text.to_string(),
            tokens_len: full_len,
        }];
    }

    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for part in split_into_sentences(text) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        let sentence_len = text_token_len(state, trimmed);
        let sentence_with_space_len = if current.is_empty() {
            sentence_len
        } else {
            let spaced = format!(" {}", trimmed);
            text_token_len(state, &spaced)
        };
        let candidate_len = current_len.saturating_add(sentence_with_space_len);

        let candidate = if current.is_empty() {
            trimmed.to_string()
        } else {
            format!("{} {}", current, trimmed)
        };

        if candidate_len <= max_text_tokens {
            current = candidate;
            current_len = candidate_len;
            continue;
        }

        if !current.is_empty() {
            sentences.push(TextChunk {
                text: std::mem::take(&mut current),
                tokens_len: current_len,
            });
            current_len = 0;
        }

        if sentence_len <= max_text_tokens {
            current = trimmed.to_string();
            current_len = sentence_len;
        } else {
            let mut remaining = trimmed.to_string();
            while !remaining.is_empty() {
                let chunk = trim_to_fit(&remaining, state, max_text_tokens)
                    .unwrap_or_else(|| remaining.clone());
                let chunk = chunk.trim();
                if chunk.is_empty() {
                    break;
                }
                let chunk_len = text_token_len(state, chunk);
                sentences.push(TextChunk {
                    text: chunk.to_string(),
                    tokens_len: chunk_len,
                });
                remaining = remaining[chunk.len()..].trim_start().to_string();
            }
            current.clear();
            current_len = 0;
        }
    }

    if !current.is_empty() {
        sentences.push(TextChunk {
            text: current,
            tokens_len: current_len,
        });
    }

    if sentences.is_empty() {
        Vec::new()
    } else {
        sentences
    }
}

fn trim_to_fit(text: &str, state: &LlamaState, max_text_tokens: usize) -> Option<String> {
    let candidate = text.trim().to_string();
    if candidate.is_empty() {
        return None;
    }

    let mut positions: Vec<usize> = candidate.char_indices().map(|(idx, _)| idx).collect();
    positions.push(candidate.len());
    let mut high = positions.len();
    let mut low = 1usize;
    let mut best = None;

    while low < high {
        let mid = (low + high) / 2;
        let slice_len = positions[mid];
        let slice = candidate[..slice_len].trim_end();
        if text_token_len(state, slice) <= max_text_tokens {
            best = Some(slice.to_string());
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    best
}

fn is_short_input_text_with_len(tokens_len: usize) -> bool {
    tokens_len <= SHORT_INPUT_TOKENS
}

fn is_single_word_input(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let mut words = trimmed.split_whitespace();
    if words.next().is_none() {
        return true;
    }

    words.next().is_none()
}

fn filter_single_word_output(output: &str) -> String {
    let first_line = output.lines().next().unwrap_or(output).trim();
    first_line.to_string()
}

fn locale_to_lang(locale: &str) -> (String, String) {
    let normalized = locale.replace('_', "-");
    let base = normalized.split('-').next().unwrap_or(normalized.as_str());

    let language = match base {
        "aa" => "Afar",
        "ab" => "Abkhazian",
        "af" => "Afrikaans",
        "ak" => "Akan",
        "am" => "Amharic",
        "an" => "Aragonese",
        "ar" => "Arabic",
        "as" => "Assamese",
        "az" => "Azerbaijani",
        "ba" => "Bashkir",
        "be" => "Belarusian",
        "bg" => "Bulgarian",
        "bm" => "Bambara",
        "bn" => "Bengali",
        "bo" => "Tibetan",
        "br" => "Breton",
        "bs" => "Bosnian",
        "ca" => "Catalan",
        "ce" => "Chechen",
        "co" => "Corsican",
        "cs" => "Czech",
        "cv" => "Chuvash",
        "cy" => "Welsh",
        "da" => "Danish",
        "de" => "German",
        "dv" => "Divehi",
        "dz" => "Dzongkha",
        "ee" => "Ewe",
        "el" => "Greek",
        "en" => "English",
        "eo" => "Esperanto",
        "es" => "Spanish",
        "et" => "Estonian",
        "eu" => "Basque",
        "fa" => "Persian",
        "ff" => "Fulah",
        "fi" => "Finnish",
        "fil" => "Filipino",
        "fo" => "Faroese",
        "fr" => "French",
        "fy" => "Western Frisian",
        "ga" => "Irish",
        "gd" => "Scottish Gaelic",
        "gl" => "Galician",
        "gn" => "Guarani",
        "gu" => "Gujarati",
        "gv" => "Manx",
        "ha" => "Hausa",
        "he" => "Hebrew",
        "hi" => "Hindi",
        "hr" => "Croatian",
        "ht" => "Haitian",
        "hu" => "Hungarian",
        "hy" => "Armenian",
        "ia" => "Interlingua",
        "id" => "Indonesian",
        "ie" => "Interlingue",
        "ig" => "Igbo",
        "ii" => "Sichuan Yi",
        "ik" => "Inupiaq",
        "io" => "Ido",
        "is" => "Icelandic",
        "it" => "Italian",
        "iu" => "Inuktitut",
        "ja" => "Japanese",
        "jv" => "Javanese",
        "ka" => "Georgian",
        "ki" => "Kikuyu",
        "kk" => "Kazakh",
        "kl" => "Kalaallisut",
        "km" => "Central Khmer",
        "kn" => "Kannada",
        "ko" => "Korean",
        "ks" => "Kashmiri",
        "ku" => "Kurdish",
        "kw" => "Cornish",
        "ky" => "Kyrgyz",
        "la" => "Latin",
        "lb" => "Luxembourgish",
        "lg" => "Ganda",
        "ln" => "Lingala",
        "lo" => "Lao",
        "lt" => "Lithuanian",
        "lu" => "Luba-Katanga",
        "lv" => "Latvian",
        "mg" => "Malagasy",
        "mi" => "Maori",
        "mk" => "Macedonian",
        "ml" => "Malayalam",
        "mn" => "Mongolian",
        "mr" => "Marathi",
        "ms" => "Malay",
        "mt" => "Maltese",
        "my" => "Burmese",
        "nb" => "Norwegian Bokmal",
        "nd" => "North Ndebele",
        "ne" => "Nepali",
        "nl" => "Dutch",
        "nn" => "Norwegian Nynorsk",
        "no" => "Norwegian",
        "nr" => "South Ndebele",
        "nv" => "Navajo",
        "ny" => "Chichewa",
        "oc" => "Occitan",
        "om" => "Oromo",
        "or" => "Oriya",
        "os" => "Ossetian",
        "pa" => "Punjabi",
        "pl" => "Polish",
        "ps" => "Pashto",
        "pt" => "Portuguese",
        "qu" => "Quechua",
        "rm" => "Romansh",
        "rn" => "Rundi",
        "ro" => "Romanian",
        "ru" => "Russian",
        "rw" => "Kinyarwanda",
        "sa" => "Sanskrit",
        "sc" => "Sardinian",
        "sd" => "Sindhi",
        "se" => "Northern Sami",
        "sg" => "Sango",
        "si" => "Sinhala",
        "sk" => "Slovak",
        "sl" => "Slovenian",
        "sn" => "Shona",
        "so" => "Somali",
        "sq" => "Albanian",
        "sr" => "Serbian",
        "ss" => "Swati",
        "st" => "Southern Sotho",
        "su" => "Sundanese",
        "sv" => "Swedish",
        "sw" => "Swahili",
        "ta" => "Tamil",
        "te" => "Telugu",
        "tg" => "Tajik",
        "th" => "Thai",
        "ti" => "Tigrinya",
        "tk" => "Turkmen",
        "tl" => "Tagalog",
        "tn" => "Tswana",
        "to" => "Tonga",
        "tr" => "Turkish",
        "ts" => "Tsonga",
        "tt" => "Tatar",
        "ug" => "Uyghur",
        "uk" => "Ukrainian",
        "ur" => "Urdu",
        "uz" => "Uzbek",
        "ve" => "Venda",
        "vi" => "Vietnamese",
        "vo" => "Volapuk",
        "wa" => "Walloon",
        "wo" => "Wolof",
        "xh" => "Xhosa",
        "yi" => "Yiddish",
        "yo" => "Yoruba",
        "za" => "Zhuang",
        "zh" => "Chinese",
        "zu" => "Zulu",
        _ => panic!("Unsupported locale: {}", locale),
    };

    (language.to_string(), normalized)
}

fn build_sampler(seed: u32) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(TOP_K),
        LlamaSampler::top_p(TOP_P, TOP_P_MIN_KEEP),
        LlamaSampler::dist(seed),
    ])
}

fn tokens_end_with(tokens: &[LlamaToken], suffix: &[LlamaToken]) -> bool {
    if suffix.is_empty() || tokens.len() < suffix.len() {
        return false;
    }

    let start = tokens.len() - suffix.len();
    tokens[start..] == *suffix
}
