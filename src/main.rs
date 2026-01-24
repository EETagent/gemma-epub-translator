mod epub_processor;
mod epub_rebuild;
mod translate;
mod ui;

use std::env;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--test" {
        run_cli_test(&args[2..]);
        return;
    }

    use cacao::appkit::App;
    App::new(
        "com.github.EETagent.gemma-translator",
        ui::app::TranslatorApp::new(),
    )
    .run();
}

fn run_cli_test(args: &[String]) {
    println!("=== Gemma Translator CLI Test ===\n");

    println!("Loading model...");
    let state = Arc::new(Mutex::new(translate::create_state()));
    println!("Model loaded!\n");

    if args.is_empty() {
        let test_texts = vec![
            "Hello, world!".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "How are you today?".to_string(),
            "This is a test of the translation system.".to_string(),
        ];

        let translated = translate::translate_texts_with_state(&state, &test_texts, "en-US", "cs");
        for (text, translated) in test_texts.iter().zip(translated.iter()) {
            println!("Original: {}", text);
            println!("Czech:    {}\n", translated);
        }
    } else {
        let text = args.join(" ");
        println!("Original: {}", text);
        let translated = translate::translate_text_with_state(&state, &text, "en-US", "cs");
        println!("Czech:    {}", translated);
    }
}
