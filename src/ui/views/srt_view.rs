use cacao::appkit::App;
use cacao::button::{BezelStyle, Button};
use cacao::color::Color;
use cacao::control::Control;
use cacao::dragdrop::{DragInfo, DragOperation};
use cacao::filesystem::FileSelectPanel;
use cacao::foundation::NSURL;
use cacao::layout::{Layout, LayoutConstraint};
use cacao::objc_access::ObjcAccess;
use cacao::pasteboard::PasteboardType;
use cacao::progress::ProgressIndicator;
use cacao::select::Select;
use cacao::text::{Font, Label, TextAlign};
use cacao::url::Url;
use cacao::view::{View, ViewDelegate};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::processors::srt_processor::{translate_srt_with_cancel, SrtProcessError};
use crate::translate::LlamaState;
use crate::ui::app::{AppMessage, TranslationKind, TranslatorApp};

struct SharedUi {
    content: View,
    label_title: Label,
    source_lang_select: Select,
    lang_select: Select,
    btn_open: Button,
    btn_translate: Button,
    btn_cancel: Button,
    progress_bar: ProgressIndicator,
    status_label: Label,
    current_path: Option<PathBuf>,
    pending_paths: VecDeque<QueuedTranslation>,
    is_translating: Arc<AtomicBool>,
    translation_started_at: Option<Instant>,
    model_state: Option<Arc<Mutex<LlamaState>>>,
}

impl SharedUi {
    fn new() -> Self {
        Self {
            content: View::default(),
            label_title: Label::default(),
            source_lang_select: Select::new(),
            lang_select: Select::new(),
            btn_open: Button::new("Open SRT..."),
            btn_translate: Button::new("Translate"),
            btn_cancel: Button::new("Cancel"),
            progress_bar: ProgressIndicator::default(),
            status_label: Label::default(),
            current_path: None,
            pending_paths: VecDeque::new(),
            is_translating: Arc::new(AtomicBool::new(false)),
            translation_started_at: None,
            model_state: None,
        }
    }
}

#[derive(Clone, Debug)]
struct QueuedTranslation {
    path: PathBuf,
    source_locale: String,
    target_locale: String,
}

pub struct SrtView {
    view: View<ContentView>,
    ui: Rc<RefCell<SharedUi>>,
}

pub struct ContentView {
    ui: Rc<RefCell<SharedUi>>,
}

const SOURCE_LANGUAGES: &[(&str, &str)] = &[
    ("US English", "en-US"),
    ("UK English", "en-GB"),
    ("German", "de"),
    ("Slovak", "sk"),
    ("Spanish", "es"),
    ("Russian", "ru"),
    ("Danish", "da"),
    ("Czech", "cs"),
];

const TARGET_LANGUAGES: &[(&str, &str)] = &[("Czech", "cs"), ("English", "en-US")];

impl ContentView {
    fn srt_paths_from_drag(info: &DragInfo) -> Vec<PathBuf> {
        let pasteboard = info.get_pasteboard();
        let Some(urls) = pasteboard.get_file_urls().ok() else {
            return Vec::new();
        };
        urls.into_iter()
            .map(|url| url.pathbuf())
            .filter(is_srt_path)
            .collect()
    }
}

impl ViewDelegate for ContentView {
    const NAME: &'static str = "SrtViewContent";

    fn did_load(&mut self, view: View) {
        view.register_for_dragged_types(&[PasteboardType::FileURL]);
        let mut ui = self.ui.borrow_mut();

        ui.content.set_background_color(Color::rgb(241, 245, 244));
        ui.label_title.set_text("Select an SRT file to begin");
        ui.label_title.set_text_alignment(TextAlign::Center);
        ui.label_title.set_font(Font::bold_system(18.0));
        ui.label_title.set_text_color(Color::Label);

        for &(name, _) in SOURCE_LANGUAGES {
            ui.source_lang_select.add_item(name);
        }
        ui.source_lang_select.set_selected_index(0);
        for &(name, _) in TARGET_LANGUAGES {
            ui.lang_select.add_item(name);
        }
        ui.lang_select.set_selected_index(0);

        ui.btn_open.set_bezel_style(BezelStyle::Rounded);
        ui.btn_open.set_action(|_| {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ShowOpenPanel(
                TranslationKind::Srt,
            ));
        });

        ui.btn_translate.set_bezel_style(BezelStyle::Rounded);
        ui.btn_translate.set_hidden(true);
        ui.btn_cancel.set_bezel_style(BezelStyle::Rounded);
        ui.btn_cancel.set_hidden(true);
        ui.btn_cancel.set_action(|_| {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::CancelTranslation(
                TranslationKind::Srt,
            ));
        });

        ui.progress_bar.set_hidden(true);
        ui.progress_bar.set_indeterminate(false);
        ui.progress_bar.set_value(0.0);
        ui.status_label.set_text("");
        ui.status_label.set_text_alignment(TextAlign::Center);
        ui.status_label.set_text_color(Color::LabelSecondary);

        ui.content.add_subview(&ui.label_title);
        ui.content.add_subview(&ui.source_lang_select);
        ui.content.add_subview(&ui.lang_select);
        ui.content.add_subview(&ui.btn_open);
        ui.content.add_subview(&ui.btn_translate);
        ui.content.add_subview(&ui.btn_cancel);
        ui.content.add_subview(&ui.progress_bar);
        ui.content.add_subview(&ui.status_label);
        view.add_subview(&ui.content);

        LayoutConstraint::activate(&[
            ui.label_title
                .top
                .constraint_equal_to(&ui.content.safe_layout_guide.top)
                .offset(64.),
            ui.label_title
                .leading
                .constraint_equal_to(&ui.content.leading)
                .offset(24.),
            ui.label_title
                .trailing
                .constraint_equal_to(&ui.content.trailing)
                .offset(-24.),
            ui.source_lang_select
                .top
                .constraint_equal_to(&ui.label_title.bottom)
                .offset(16.),
            ui.source_lang_select
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.lang_select
                .top
                .constraint_equal_to(&ui.source_lang_select.bottom)
                .offset(10.),
            ui.lang_select
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.btn_open
                .top
                .constraint_equal_to(&ui.lang_select.bottom)
                .offset(20.),
            ui.btn_open
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.btn_translate
                .top
                .constraint_equal_to(&ui.btn_open.bottom)
                .offset(10.),
            ui.btn_translate
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.btn_cancel
                .top
                .constraint_equal_to(&ui.btn_translate.bottom)
                .offset(10.),
            ui.btn_cancel
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.progress_bar
                .top
                .constraint_equal_to(&ui.btn_cancel.bottom)
                .offset(24.),
            ui.progress_bar.width.constraint_equal_to_constant(240.),
            ui.progress_bar
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.status_label
                .top
                .constraint_equal_to(&ui.progress_bar.bottom)
                .offset(10.),
            ui.status_label
                .leading
                .constraint_equal_to(&ui.content.leading)
                .offset(24.),
            ui.status_label
                .trailing
                .constraint_equal_to(&ui.content.trailing)
                .offset(-24.),
            ui.content
                .top
                .constraint_equal_to(&view.safe_layout_guide.top),
            ui.content
                .leading
                .constraint_equal_to(&view.safe_layout_guide.leading),
            ui.content
                .trailing
                .constraint_equal_to(&view.safe_layout_guide.trailing),
            ui.content
                .bottom
                .constraint_equal_to(&view.safe_layout_guide.bottom),
        ]);
    }

    fn dragging_entered(&self, info: DragInfo) -> DragOperation {
        if Self::srt_paths_from_drag(&info).is_empty() {
            DragOperation::None
        } else {
            DragOperation::Copy
        }
    }

    fn prepare_for_drag_operation(&self, info: DragInfo) -> bool {
        !Self::srt_paths_from_drag(&info).is_empty()
    }

    fn perform_drag_operation(&self, info: DragInfo) -> bool {
        let paths = Self::srt_paths_from_drag(&info);
        if paths.is_empty() {
            return false;
        }
        for path in paths {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile {
                kind: TranslationKind::Srt,
                path,
            });
        }
        true
    }
}

impl SrtView {
    pub fn new() -> Self {
        let ui = Rc::new(RefCell::new(SharedUi::new()));
        Self {
            view: View::with(ContentView { ui: ui.clone() }),
            ui,
        }
    }

    pub fn set_model_state(&self, state: Arc<Mutex<LlamaState>>) {
        let mut ui = self.ui.borrow_mut();
        ui.model_state = Some(state);
        ui.status_label
            .set_text("Model loaded. Ready to translate SRT.");
        ui.btn_translate.set_enabled(true);
    }

    pub fn clear_model_state(&self) {
        let mut ui = self.ui.borrow_mut();
        ui.model_state = None;
    }

    pub fn set_loading_model(&self) {
        let ui = self.ui.borrow();
        ui.status_label.set_text("Loading model...");
        ui.btn_translate.set_enabled(false);
    }

    pub fn set_model_load_error(&self, error: &str) {
        let ui = self.ui.borrow();
        ui.status_label
            .set_text(&format!("Failed to load model: {}", error));
        ui.btn_translate.set_enabled(false);
    }

    pub fn view_id(&self) -> cacao::foundation::id {
        self.view
            .get_from_backing_obj(|obj| obj as *const _ as cacao::foundation::id)
    }

    pub fn is_translating_flag(&self) -> Arc<AtomicBool> {
        self.ui.borrow().is_translating.clone()
    }

    pub fn present_open_panel(&self) {
        let mut panel = FileSelectPanel::new();
        panel.set_message("Choose SRT subtitle files");
        panel.set_allows_multiple_selection(true);
        panel.set_can_choose_directories(false);
        panel.set_can_choose_files(true);
        panel.show(|urls| {
            let paths = srt_paths_from_nsurls(&urls);
            for path in paths {
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile {
                    kind: TranslationKind::Srt,
                    path,
                });
            }
        });
    }

    pub fn open_urls(&self, urls: Vec<Url>) {
        for path in srt_paths_from_urls(&urls) {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile {
                kind: TranslationKind::Srt,
                path,
            });
        }
    }

    pub fn handle_open_file(&self, path: PathBuf) {
        if !is_srt_path(&path) {
            self.ui
                .borrow()
                .status_label
                .set_text("Please select an .srt file.");
            return;
        }

        let should_queue = self.ui.borrow().is_translating.load(Ordering::SeqCst);
        if should_queue {
            let mut ui = self.ui.borrow_mut();
            let (source_locale, target_locale) = selected_locales(&ui);
            ui.pending_paths.push_back(QueuedTranslation {
                path,
                source_locale,
                target_locale,
            });
            ui.status_label
                .set_text(&format!("Queued {} SRT file(s).", ui.pending_paths.len()));
            return;
        }

        let (source_locale, target_locale) = {
            let ui = self.ui.borrow();
            selected_locales(&ui)
        };
        self.start_translation_for_path(path, source_locale, target_locale);
    }

    pub fn handle_progress(&self, completed: usize, total: usize) {
        let ui = self.ui.borrow_mut();
        let percentage = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let eta = if completed >= 5 {
            let elapsed = ui
                .translation_started_at
                .map(|start| start.elapsed().as_secs_f64())
                .unwrap_or_default();
            let per_unit = elapsed / completed as f64;
            let remaining = (total.saturating_sub(completed)) as f64 * per_unit;
            format!(" — ~{} min left", (remaining / 60.0).ceil().max(1.0) as u64)
        } else {
            String::new()
        };
        ui.progress_bar.set_value(percentage);
        ui.status_label.set_text(&format!(
            "Translating... {}/{} subtitle lines ({:.0}%){}",
            completed, total, percentage, eta
        ));
    }

    pub fn handle_completion(&self, result: Result<PathBuf, String>) {
        let next = {
            let mut ui = self.ui.borrow_mut();
            ui.is_translating.store(false, Ordering::SeqCst);
            ui.btn_cancel.set_hidden(true);
            ui.progress_bar.set_value(100.0);
            ui.translation_started_at = None;
            match result {
                Ok(path) => ui
                    .status_label
                    .set_text(&format!("✓ Saved to {}", path.display())),
                Err(err) => ui.status_label.set_text(&format!("✗ Error: {}", err)),
            }
            ui.pending_paths.pop_front()
        };

        if let Some(next) = next {
            self.start_translation_for_path(next.path, next.source_locale, next.target_locale);
        }
    }

    pub fn handle_cancel(&self) {
        let mut ui = self.ui.borrow_mut();
        ui.is_translating.store(false, Ordering::SeqCst);
        ui.btn_cancel.set_hidden(true);
        ui.progress_bar.set_hidden(true);
        ui.status_label.set_text("Translation cancelled.");
        ui.pending_paths.clear();
        ui.translation_started_at = None;
    }

    fn start_translation_for_path(
        &self,
        path: PathBuf,
        source_locale: String,
        target_locale: String,
    ) {
        let mut ui = self.ui.borrow_mut();
        ui.current_path = Some(path.clone());
        let name = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("subtitle.srt");
        ui.label_title.set_text(&format!("Translating {name}"));
        ui.progress_bar.set_hidden(false);
        ui.progress_bar.set_value(0.0);
        ui.btn_translate.set_hidden(true);
        ui.btn_cancel.set_hidden(false);
        ui.is_translating.store(true, Ordering::SeqCst);
        ui.translation_started_at = Some(Instant::now());

        let Some(model_state) = ui.model_state.clone() else {
            ui.status_label
                .set_text("Model not loaded yet. Please wait...");
            ui.is_translating.store(false, Ordering::SeqCst);
            ui.btn_cancel.set_hidden(true);
            return;
        };
        drop(ui);

        let is_translating = self.ui.borrow().is_translating.clone();
        let is_translating_for_progress = is_translating.clone();
        std::thread::spawn(move || {
            let cancel_flag = is_translating.clone();
            let result = translate_srt_with_cancel(
                &model_state,
                &path,
                &source_locale,
                &target_locale,
                move |done, total| {
                    if !is_translating_for_progress.load(Ordering::SeqCst) {
                        return;
                    }
                    App::<TranslatorApp, AppMessage>::dispatch_main(
                        AppMessage::TranslationProgress {
                            kind: TranslationKind::Srt,
                            completed: done,
                            total,
                        },
                    );
                },
                Some(cancel_flag),
            );

            if !is_translating.load(Ordering::SeqCst) {
                return;
            }

            let message = match result {
                Ok(path) => AppMessage::TranslationComplete {
                    kind: TranslationKind::Srt,
                    result: Ok(path),
                },
                Err(SrtProcessError::Cancelled) => return,
                Err(err) => AppMessage::TranslationComplete {
                    kind: TranslationKind::Srt,
                    result: Err(format!("{:?}", err)),
                },
            };
            App::<TranslatorApp, AppMessage>::dispatch_main(message);
        });
    }
}

fn is_srt_path(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

fn srt_paths_from_urls(urls: &[Url]) -> Vec<PathBuf> {
    urls.iter()
        .filter_map(|url| url.to_file_path().ok())
        .filter(is_srt_path)
        .collect()
}

fn srt_paths_from_nsurls(urls: &[NSURL]) -> Vec<PathBuf> {
    urls.iter()
        .map(|url| url.pathbuf())
        .filter(is_srt_path)
        .collect()
}

fn selected_locales(ui: &SharedUi) -> (String, String) {
    let source = SOURCE_LANGUAGES
        .get(ui.source_lang_select.get_selected_index())
        .map(|&(_, code)| code)
        .unwrap_or("en-US");
    let target = TARGET_LANGUAGES
        .get(ui.lang_select.get_selected_index())
        .map(|&(_, code)| code)
        .unwrap_or("cs");
    (source.to_string(), target.to_string())
}
