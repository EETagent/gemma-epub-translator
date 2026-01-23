use cacao::appkit::App;
use cacao::button::{BezelStyle, Button};
use cacao::color::Color;
use cacao::dragdrop::{DragInfo, DragOperation};
use cacao::filesystem::FileSelectPanel;
use cacao::foundation::NSURL;
use cacao::image::{Image, ImageView};
use cacao::layout::Layout;
use cacao::layout::LayoutConstraint;
use cacao::objc_access::ObjcAccess;
use cacao::pasteboard::PasteboardType;
use cacao::progress::ProgressIndicator;
use cacao::select::Select;
use cacao::text::{Font, Label, TextAlign};
use cacao::url::Url;
use cacao::view::{View, ViewDelegate};
use epub::doc::EpubDoc;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::epub_processor::{extract_text_segments, translate_epub_with_progress};
use crate::ui::app::{AppMessage, TranslatorApp};

struct SharedUi {
    content: View,
    label_title: Label,
    img_cover: ImageView,
    source_lang_select: Select,
    lang_select: Select,
    btn_open: Button,
    btn_translate: Button,
    btn_cancel: Button,
    progress_bar: ProgressIndicator,
    status_label: Label,
    cover_image: Option<Image>,
    current_path: Option<PathBuf>,
    pending_paths: VecDeque<QueuedTranslation>,
    is_translating: Arc<AtomicBool>,
    translation_started_at: Option<Instant>,
}

impl SharedUi {
    fn new() -> Self {
        Self {
            content: View::default(),
            label_title: Label::default(),
            img_cover: ImageView::default(),
            source_lang_select: Select::new(),
            lang_select: Select::new(),
            btn_open: Button::new("Open EPUB..."),
            btn_translate: Button::new("Translate"),
            btn_cancel: Button::new("Cancel"),
            progress_bar: ProgressIndicator::default(),
            status_label: Label::default(),
            cover_image: None,
            current_path: None,
            pending_paths: VecDeque::new(),
            is_translating: Arc::new(AtomicBool::new(false)),
            translation_started_at: None,
        }
    }
}

#[derive(Clone, Debug)]
struct QueuedTranslation {
    path: PathBuf,
    source_locale: String,
    target_locale: String,
}

pub struct EpubView {
    view: View<ContentView>,
    ui: Rc<RefCell<SharedUi>>,
}

pub struct ContentView {
    ui: Rc<RefCell<SharedUi>>,
}

impl ContentView {
    fn epub_paths_from_drag(info: &DragInfo) -> Vec<PathBuf> {
        let pasteboard = info.get_pasteboard();
        let Some(urls) = pasteboard.get_file_urls().ok() else {
            return Vec::new();
        };
        urls.into_iter()
            .map(|url| url.pathbuf())
            .filter(is_epub_path)
            .collect()
    }
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

impl ViewDelegate for ContentView {
    const NAME: &'static str = "EpubViewContent";

    fn did_load(&mut self, view: View) {
        view.register_for_dragged_types(&[PasteboardType::FileURL]);

        let mut ui = self.ui.borrow_mut();

        ui.content.set_background_color(Color::rgb(245, 244, 241));

        ui.img_cover.layer.set_corner_radius(12.0);

        ui.label_title.set_text("Select an EPUB to begin");
        ui.label_title.set_text_alignment(TextAlign::Center);
        ui.label_title.set_font(Font::bold_system(18.0));
        ui.label_title.set_text_color(Color::Label);

        for &(lang_name, _lang_code) in SOURCE_LANGUAGES {
            ui.source_lang_select.add_item(lang_name);
        }
        ui.source_lang_select.set_selected_index(0);

        for &(lang_name, _lang_code) in TARGET_LANGUAGES {
            ui.lang_select.add_item(lang_name);
        }
        ui.lang_select.set_selected_index(0);

        ui.btn_open.set_bezel_style(BezelStyle::Rounded);
        ui.btn_open.set_action(|_| {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ShowOpenPanel);
        });

        ui.btn_translate.set_bezel_style(BezelStyle::Rounded);
        ui.btn_translate.set_hidden(true);
        ui.btn_translate.set_action(|_| {
            // Will be handled by starting translation
        });

        ui.btn_cancel.set_bezel_style(BezelStyle::Rounded);
        ui.btn_cancel.set_hidden(true);
        ui.btn_cancel.set_action(|_| {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::CancelTranslation);
        });

        ui.progress_bar.set_hidden(true);
        ui.progress_bar.set_indeterminate(false);
        ui.progress_bar.set_value(0.0);

        ui.status_label.set_text("");
        ui.status_label.set_text_alignment(TextAlign::Center);
        ui.status_label.set_text_color(Color::LabelSecondary);

        ui.content.add_subview(&ui.img_cover);
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
            ui.img_cover
                .top
                .constraint_equal_to(&ui.content.safe_layout_guide.top)
                .offset(32.),
            ui.img_cover
                .center_x
                .constraint_equal_to(&ui.content.center_x),
            ui.img_cover.width.constraint_equal_to_constant(180.),
            ui.img_cover.height.constraint_equal_to_constant(240.),
            ui.label_title
                .top
                .constraint_equal_to(&ui.img_cover.bottom)
                .offset(18.),
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
                .offset(12.),
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
        ]);

        LayoutConstraint::activate(&[
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
        if Self::epub_paths_from_drag(&info).is_empty() {
            DragOperation::None
        } else {
            DragOperation::Copy
        }
    }

    fn prepare_for_drag_operation(&self, info: DragInfo) -> bool {
        !Self::epub_paths_from_drag(&info).is_empty()
    }

    fn perform_drag_operation(&self, info: DragInfo) -> bool {
        let paths = Self::epub_paths_from_drag(&info);
        if paths.is_empty() {
            false
        } else {
            for path in paths {
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
            }
            true
        }
    }
}

impl EpubView {
    pub fn new() -> Self {
        let ui = Rc::new(RefCell::new(SharedUi::new()));
        Self {
            view: View::with(ContentView { ui: ui.clone() }),
            ui,
        }
    }

    pub fn view(&self) -> &View<ContentView> {
        &self.view
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
        panel.set_message("Choose an EPUB file");
        panel.set_allows_multiple_selection(true);
        panel.set_can_choose_directories(false);
        panel.set_can_choose_files(true);
        panel.show(|urls| {
            let paths = epub_paths_from_nsurls(&urls);
            for path in paths {
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
            }
        });
    }

    pub fn handle_open_file(&self, path: PathBuf) {
        if !is_epub_path(&path) {
            let mut ui = self.ui.borrow_mut();
            ui.status_label.set_text("Please select an .epub file.");
            return;
        }

        let should_queue = {
            let ui = self.ui.borrow();
            ui.is_translating.load(Ordering::SeqCst)
        };

        if should_queue {
            let mut ui = self.ui.borrow_mut();
            let (source_locale, target_locale) = selected_locales(&ui);
            ui.pending_paths.push_back(QueuedTranslation {
                path,
                source_locale,
                target_locale,
            });
            let queued = ui.pending_paths.len();
            let suffix = if queued == 1 { "" } else { "s" };
            ui.status_label
                .set_text(&format!("Queued {queued} EPUB{suffix}."));
            return;
        }

        let (source_locale, target_locale) = {
            let ui = self.ui.borrow();
            selected_locales(&ui)
        };
        self.start_translation_for_path(path, source_locale, target_locale);
    }

    pub fn open_urls(&self, urls: Vec<Url>) {
        for path in epub_paths_from_urls(&urls) {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
        }
    }

    pub fn handle_progress(&self, completed: usize, total: usize) {
        let mut ui = self.ui.borrow_mut();
        let queued = ui.pending_paths.len();
        let percentage = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let remaining_text = if completed >= 5 {
            let elapsed = ui
                .translation_started_at
                .map(|start| start.elapsed().as_secs_f64())
                .unwrap_or_default();
            let per_segment = elapsed / completed as f64;
            let remaining = (total.saturating_sub(completed)) as f64 * per_segment;
            let minutes = (remaining / 60.0).ceil() as u64;
            format!(" — ~{} min left", minutes.max(1))
        } else {
            String::new()
        };
        ui.progress_bar.set_value(percentage);
        let queued_text = if queued > 0 {
            format!(" — {queued} queued")
        } else {
            String::new()
        };
        ui.status_label.set_text(&format!(
            "Translating... {}/{} segments ({:.0}%){}{}",
            completed, total, percentage, remaining_text, queued_text
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
                Ok(path) => {
                    ui.status_label
                        .set_text(&format!("✓ Saved to {}", path.display()));
                }
                Err(err) => {
                    ui.status_label.set_text(&format!("✗ Error: {}", err));
                }
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
        let mut opened = true;

        match EpubDoc::new(&path) {
            Ok(mut doc) => {
                let title = doc
                    .get_title()
                    .unwrap_or_else(|| "Untitled EPUB".to_string());
                ui.label_title.set_text(&title);

                if let Some((cover_data, _mime)) = doc.get_cover() {
                    let image = Image::with_data(&cover_data);
                    ui.img_cover.set_image(&image);
                    ui.cover_image = Some(image);
                    ui.img_cover.set_background_color(Color::Clear);
                } else {
                    ui.img_cover.set_background_color(Color::rgb(220, 220, 220));
                }

                ui.current_path = Some(path.clone());
            }
            Err(err) => {
                ui.status_label
                    .set_text(&format!("Failed to open EPUB: {err}"));
                opened = false;
            }
        }

        drop(ui);

        if opened {
            self.analyze_and_start_translation(&path, source_locale, target_locale);
        } else {
            self.start_next_from_queue();
        }
    }

    fn start_next_from_queue(&self) {
        let next = {
            let mut ui = self.ui.borrow_mut();
            ui.pending_paths.pop_front()
        };

        if let Some(next) = next {
            self.start_translation_for_path(next.path, next.source_locale, next.target_locale);
        }
    }

    fn analyze_and_start_translation(
        &self,
        path: &PathBuf,
        source_locale: String,
        target_locale: String,
    ) {
        let segments = match extract_text_segments(path) {
            Ok(result) => result,
            Err(err) => {
                let mut ui = self.ui.borrow_mut();
                ui.status_label
                    .set_text(&format!("Failed to parse EPUB: {err:?}"));
                return;
            }
        };

        let segment_count = segments.segments.len();
        let word_count = segments.total_words;

        {
            let mut ui = self.ui.borrow_mut();
            ui.status_label.set_text(&format!(
                "Found {} segments ({} words). Starting translation...",
                segment_count, word_count
            ));
            ui.progress_bar.set_hidden(false);
            ui.progress_bar.set_value(0.0);
            ui.btn_translate.set_hidden(true);
            ui.btn_cancel.set_hidden(false);
            ui.is_translating.store(true, Ordering::SeqCst);
            ui.translation_started_at = Some(Instant::now());
        }

        let path = path.clone();
        let is_translating = self.ui.borrow().is_translating.clone();

        std::thread::spawn(move || {
            let completed = Arc::new(AtomicUsize::new(0));
            let total = Arc::new(AtomicUsize::new(segment_count));
            let completed_clone = completed.clone();
            let total_clone = total.clone();
            let is_translating_clone = is_translating.clone();

            let result = translate_epub_with_progress(
                &path,
                &source_locale,
                &target_locale,
                move |done, t| {
                    if !is_translating_clone.load(Ordering::SeqCst) {
                        return;
                    }

                    completed_clone.store(done, Ordering::SeqCst);
                    total_clone.store(t, Ordering::SeqCst);

                    App::<TranslatorApp, AppMessage>::dispatch_main(
                        AppMessage::TranslationProgress {
                            completed: done,
                            total: t,
                        },
                    );
                },
            );

            if !is_translating.load(Ordering::SeqCst) {
                return;
            }

            let message = match result {
                Ok(path) => AppMessage::TranslationComplete(Ok(path)),
                Err(err) => AppMessage::TranslationComplete(Err(format!("{:?}", err))),
            };
            App::<TranslatorApp, AppMessage>::dispatch_main(message);
        });
    }
}

fn is_epub_path(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("epub"))
        .unwrap_or(false)
}

fn epub_paths_from_urls(urls: &[Url]) -> Vec<PathBuf> {
    urls.iter()
        .filter_map(|url| url.to_file_path().ok())
        .filter(is_epub_path)
        .collect()
}

fn epub_paths_from_nsurls(urls: &[NSURL]) -> Vec<PathBuf> {
    urls.iter()
        .map(|url| url.pathbuf())
        .filter(is_epub_path)
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
