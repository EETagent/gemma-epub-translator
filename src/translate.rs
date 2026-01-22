use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use std::env;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

// translategemma-4b-it-q8_0.gguf
const MODEL_FILE: &str = "translategemma-12b-it.Q4_K_M.gguf";
const CTX_SIZE: u32 = 2048 * 4;
const N_BATCH: u32 = 2048 * 4;
const MAX_OUTPUT_TOKENS: usize = 512 * 4;
const MAX_SEQ_BATCH: usize = 8;

const PROMPT_TEMPLATE: &str = "<bos><start_of_turn>user\nYou are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.\nProduce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:\n\n\n{TEXT}<end_of_turn>\n<start_of_turn>model\n";

static MODEL_STATE: OnceLock<Arc<Mutex<LlamaState>>> = OnceLock::new();

struct LlamaState {
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: &'static LlamaModel,
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    max_seq_batch: usize,
}

unsafe impl Send for LlamaState {}

pub fn init_model() {
    let _ = get_model_state();
}

pub fn translate_text(text: &str, source_locale: &str, target_locale: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let results = translate_texts(&[text.to_string()], source_locale, target_locale);
    results
        .into_iter()
        .next()
        .unwrap_or_else(|| text.to_string())
}

pub struct BatchLimits {
    pub max_prompt_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_sequences: usize,
}

#[derive(Clone, Copy, Debug)]
struct PromptKey {
    parent_index: usize,
    chunk_index: usize,
}

pub fn translate_texts(texts: &[String], source_locale: &str, target_locale: &str) -> Vec<String> {
    if texts.is_empty() {
        return Vec::new();
    }

    let state = get_model_state();
    let mut state = state.lock().expect("llama state lock");
    let limits = BatchLimits {
        max_prompt_tokens: CTX_SIZE as usize - MAX_OUTPUT_TOKENS,
        max_batch_tokens: state.ctx.n_batch() as usize,
        max_sequences: state.max_seq_batch,
    };
    let mut outputs = vec![String::new(); texts.len()];
    let mut prompts = Vec::new();
    let mut chunk_results: Vec<Vec<Option<String>>> = vec![Vec::new(); texts.len()];

    for (index, text) in texts.iter().enumerate() {
        if text.trim().is_empty() {
            outputs[index] = text.to_string();
            continue;
        }

        let chunks = split_text_to_fit(text, source_locale, target_locale, &state, &limits);
        if chunks.is_empty() {
            outputs[index] = text.to_string();
            continue;
        }

        chunk_results[index] = vec![None; chunks.len()];
        for (chunk_index, chunk_text) in chunks.into_iter().enumerate() {
            let prompt = build_prompt(&chunk_text, source_locale, target_locale);
            prompts.push((
                PromptKey {
                    parent_index: index,
                    chunk_index,
                },
                prompt,
            ));
        }
    }

    if prompts.is_empty() {
        return outputs;
    }

    match run_batch_inference(&mut state, &prompts) {
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

pub fn batch_limits() -> BatchLimits {
    let state = get_model_state();
    let state = state.lock().expect("llama state lock");

    BatchLimits {
        max_prompt_tokens: CTX_SIZE as usize - MAX_OUTPUT_TOKENS,
        max_batch_tokens: state.ctx.n_batch() as usize,
        max_sequences: state.max_seq_batch,
    }
}

pub fn prompt_token_count(text: &str, source_locale: &str, target_locale: &str) -> usize {
    if text.trim().is_empty() {
        return 0;
    }

    let prompt = build_prompt(text, source_locale, target_locale);
    let state = get_model_state();
    let state = state.lock().expect("llama state lock");

    prompt_token_len(&state, &prompt)
}

pub fn is_model_ready() -> bool {
    MODEL_STATE.get().is_some()
}

fn get_model_state() -> Arc<Mutex<LlamaState>> {
    MODEL_STATE
        .get_or_init(|| Arc::new(Mutex::new(init_state())))
        .clone()
}

fn run_batch_inference(
    state: &mut LlamaState,
    prompts: &[(PromptKey, String)],
) -> Result<Vec<(PromptKey, String)>, llama_cpp_2::LlamaCppError> {
    let max_batch = state.ctx.n_batch() as usize;
    let max_sequences = state.max_seq_batch;
    let mut results = Vec::with_capacity(prompts.len());
    let mut index = 0;

    while index < prompts.len() {
        let mut batch_indices = Vec::new();
        let mut batch_tokens = Vec::new();
        let mut token_count = 0usize;

        while index < prompts.len() {
            let tokens = state
                .model
                .str_to_token(&prompts[index].1, AddBos::Never)
                .unwrap_or_default();
            if tokens.is_empty() {
                results.push((prompts[index].0, String::new()));
                index += 1;
                continue;
            }

            let token_len = tokens.len();
            if batch_tokens.len() >= max_sequences {
                break;
            }

            if !batch_tokens.is_empty() && token_count + token_len > max_batch {
                break;
            }

            token_count += token_len;
            batch_indices.push(prompts[index].0);
            batch_tokens.push(tokens);
            index += 1;
        }

        if batch_tokens.is_empty() {
            continue;
        }

        let outputs = run_batch_with_tokens(state, &batch_tokens)?;
        for (prompt_index, output) in batch_indices.into_iter().zip(outputs.into_iter()) {
            results.push((prompt_index, output));
        }
    }

    Ok(results)
}

fn run_batch_with_tokens(
    state: &mut LlamaState,
    prompt_tokens: &[Vec<LlamaToken>],
) -> Result<Vec<String>, llama_cpp_2::LlamaCppError> {
    state.ctx.clear_kv_cache();

    let total_tokens: usize = prompt_tokens.iter().map(|tokens| tokens.len()).sum();
    let seq_count = prompt_tokens.len();

    let mut batch = LlamaBatch::new(total_tokens, seq_count as i32);
    let mut logits_indices = Vec::with_capacity(seq_count);
    let mut offset = 0usize;

    for (seq_id, tokens) in prompt_tokens.iter().enumerate() {
        batch.add_sequence(tokens, seq_id as i32, false).ok();
        let last_offset = offset + tokens.len().saturating_sub(1);
        logits_indices.push(last_offset as i32);
        offset += tokens.len();
    }

    state.ctx.decode(&mut batch).ok();

    let mut output_tokens = vec![Vec::with_capacity(MAX_OUTPUT_TOKENS); seq_count];
    let mut positions: Vec<i32> = prompt_tokens
        .iter()
        .map(|tokens| tokens.len() as i32)
        .collect();
    let mut active = vec![true; seq_count];

    for _ in 0..MAX_OUTPUT_TOKENS {
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

            let token = data_array.sample_token_greedy();
            if state.model.is_eog_token(token) {
                active[seq_index] = false;
                continue;
            }

            output_tokens[seq_index].push(token);
            next_tokens.push(token);
            active_indices.push(seq_index);
        }

        if active_indices.is_empty() {
            break;
        }

        let mut token_batch = LlamaBatch::new(active_indices.len(), seq_count as i32);
        for (batch_index, seq_index) in active_indices.iter().enumerate() {
            let pos = positions[*seq_index];
            token_batch
                .add(next_tokens[batch_index], pos, &[*seq_index as i32], true)
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
    let model = Box::leak(model);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CTX_SIZE))
        .with_n_batch(N_BATCH)
        .with_n_seq_max(MAX_SEQ_BATCH as u32);

    let ctx = model
        .new_context(&backend, ctx_params)
        .expect("init context");

    LlamaState {
        backend,
        model,
        ctx,
        max_seq_batch: MAX_SEQ_BATCH,
    }
}

fn build_prompt(text: &str, source_locale: &str, target_locale: &str) -> String {
    let (source_lang, source_code) = locale_to_lang(source_locale);
    let (target_lang, target_code) = locale_to_lang(target_locale);

    PROMPT_TEMPLATE
        .replace("{SOURCE_LANG}", source_lang.as_str())
        .replace("{SOURCE_CODE}", source_code.as_str())
        .replace("{TARGET_LANG}", target_lang.as_str())
        .replace("{TARGET_CODE}", target_code.as_str())
        .replace("{TEXT}", text)
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

fn prompt_token_len(state: &LlamaState, prompt: &str) -> usize {
    state
        .model
        .str_to_token(prompt, AddBos::Never)
        .unwrap_or_default()
        .len()
}

fn split_text_to_fit(
    text: &str,
    source_locale: &str,
    target_locale: &str,
    state: &LlamaState,
    limits: &BatchLimits,
) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let prompt = build_prompt(text, source_locale, target_locale);
    if prompt_token_len(state, &prompt) <= limits.max_prompt_tokens {
        return vec![text.to_string()];
    }

    let mut sentences = Vec::new();
    let mut current = String::new();

    for part in split_into_sentences(text) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        let candidate = if current.is_empty() {
            trimmed.to_string()
        } else {
            format!("{} {}", current, trimmed)
        };

        let prompt = build_prompt(&candidate, source_locale, target_locale);
        if prompt_token_len(state, &prompt) <= limits.max_prompt_tokens {
            current = candidate;
            continue;
        }

        if !current.is_empty() {
            sentences.push(std::mem::take(&mut current));
        }

        let standalone = trimmed.to_string();
        let prompt = build_prompt(&standalone, source_locale, target_locale);
        if prompt_token_len(state, &prompt) <= limits.max_prompt_tokens {
            current = standalone;
        } else {
            let mut remaining = trimmed.to_string();
            while !remaining.is_empty() {
                let chunk = trim_to_fit(&remaining, source_locale, target_locale, state, limits)
                    .unwrap_or_else(|| remaining.clone());
                let chunk = chunk.trim();
                if chunk.is_empty() {
                    break;
                }
                sentences.push(chunk.to_string());
                remaining = remaining[chunk.len()..].trim_start().to_string();
            }
            current.clear();
        }
    }

    if !current.is_empty() {
        sentences.push(current);
    }

    if sentences.is_empty() {
        Vec::new()
    } else {
        sentences
    }
}

fn trim_to_fit(
    text: &str,
    source_locale: &str,
    target_locale: &str,
    state: &LlamaState,
    limits: &BatchLimits,
) -> Option<String> {
    let candidate = text.trim().to_string();
    if candidate.is_empty() {
        return None;
    }

    let mut high = candidate.len();
    let mut low = 0usize;
    let mut best = None;

    while low < high {
        let mid = (low + high) / 2;
        let slice = candidate[..mid].trim_end();
        let prompt = build_prompt(slice, source_locale, target_locale);
        if prompt_token_len(state, &prompt) <= limits.max_prompt_tokens {
            best = Some(slice.to_string());
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    best
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
