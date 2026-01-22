use cacao::appkit::menu::{Menu, MenuItem};
use cacao::appkit::window::Window;
use cacao::appkit::{App, AppDelegate};
use cacao::button::{BezelStyle, Button};
use cacao::color::Color;
use cacao::dragdrop::{DragInfo, DragOperation};
use cacao::events::EventModifierFlag;
use cacao::filesystem::FileSelectPanel;
use cacao::foundation::NSURL;
use cacao::image::{Image, ImageView};
use cacao::layout::{Layout, LayoutConstraint};
use cacao::notification_center::Dispatcher;
use cacao::pasteboard::PasteboardType;
use cacao::progress::ProgressIndicator;
use cacao::select::Select;
use cacao::text::{Font, Label, TextAlign};
use cacao::url::Url;
use cacao::view::{View, ViewDelegate};
use epub::doc::EpubDoc;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::epub_processor::{extract_text_segments, translate_epub_with_progress};
use crate::translate::init_model;

#[derive(Debug)]
pub enum AppMessage {
    ShowOpenPanel,
    OpenFile(PathBuf),
    TranslationProgress { completed: usize, total: usize },
    TranslationComplete(Result<PathBuf, String>),
    CancelTranslation,
}

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
            is_translating: Arc::new(AtomicBool::new(false)),
            translation_started_at: None,
        }
    }
}

pub struct TranslatorApp {
    window: Window,
    content_view: View<ContentView>,
    ui: Rc<RefCell<SharedUi>>,
}

struct ContentView {
    ui: Rc<RefCell<SharedUi>>,
}

impl ContentView {
    fn first_epub_path_from_drag(info: &DragInfo) -> Option<PathBuf> {
        let pasteboard = info.get_pasteboard();
        let urls = pasteboard.get_file_urls().ok()?;
        urls.into_iter().map(|url| url.pathbuf()).find(is_epub_path)
    }
}

impl ViewDelegate for ContentView {
    const NAME: &'static str = "TranslatorContentView";

    fn did_load(&mut self, view: View) {
        view.register_for_dragged_types(&[PasteboardType::FileURL]);

        let mut ui = self.ui.borrow_mut();

        ui.content.set_background_color(Color::rgb(245, 244, 241));

        ui.img_cover.layer.set_corner_radius(12.0);

        ui.label_title.set_text("Select an EPUB to begin");
        ui.label_title.set_text_alignment(TextAlign::Center);
        ui.label_title.set_font(Font::bold_system(18.0));
        ui.label_title.set_text_color(Color::Label);

        ui.source_lang_select.add_item("US English (en-US)");
        ui.source_lang_select.add_item("UK English (en-GB)");
        ui.source_lang_select.add_item("Czech (cs)");
        ui.source_lang_select.add_item("Spanish (es)");
        ui.source_lang_select.add_item("Russian (ru)");
        ui.source_lang_select.add_item("German (de)");
        ui.source_lang_select.add_item("Danish (da)");

        ui.source_lang_select.set_selected_index(0);

        ui.lang_select.add_item("Czech (cs)");
        ui.lang_select.add_item("English (en-US)");
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
        if Self::first_epub_path_from_drag(&info).is_some() {
            DragOperation::Copy
        } else {
            DragOperation::None
        }
    }

    fn prepare_for_drag_operation(&self, info: DragInfo) -> bool {
        Self::first_epub_path_from_drag(&info).is_some()
    }

    fn perform_drag_operation(&self, info: DragInfo) -> bool {
        if let Some(path) = Self::first_epub_path_from_drag(&info) {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
            true
        } else {
            false
        }
    }
}

impl TranslatorApp {
    pub fn new() -> Self {
        let ui = Rc::new(RefCell::new(SharedUi::new()));
        Self {
            window: Window::default(),
            content_view: View::with(ContentView { ui: ui.clone() }),
            ui,
        }
    }

    fn present_open_panel(&self) {
        let mut panel = FileSelectPanel::new();
        panel.set_message("Choose an EPUB file");
        panel.set_allows_multiple_selection(false);
        panel.set_can_choose_directories(false);
        panel.set_can_choose_files(true);
        panel.show(|urls| {
            if let Some(path) = first_epub_path_from_nsurls(&urls) {
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
            }
        });
    }

    fn handle_open_file(&self, path: PathBuf) {
        if !is_epub_path(&path) {
            let mut ui = self.ui.borrow_mut();
            ui.status_label.set_text("Please select an .epub file.");
            return;
        }

        let mut ui = self.ui.borrow_mut();
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

                drop(ui);
                self.analyze_and_start_translation(&path);
            }
            Err(err) => {
                ui.status_label
                    .set_text(&format!("Failed to open EPUB: {err}"));
            }
        }
    }

    fn analyze_and_start_translation(&self, path: &PathBuf) {
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

        let (source_locale, target_locale) = {
            let ui = self.ui.borrow();
            let source = match ui.source_lang_select.get_selected_index() {
                0 => "en-US",
                1 => "en-GB",
                2 => "es",
                3 => "ru",
                4 => "de",
                5 => "da",
                _ => "en-US",
            };
            let target = match ui.lang_select.get_selected_index() {
                0 => "cs",
                _ => "cs",
            };
            (source.to_string(), target.to_string())
        };

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

    fn handle_progress(&self, completed: usize, total: usize) {
        let mut ui = self.ui.borrow_mut();
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
        ui.status_label.set_text(&format!(
            "Translating... {}/{} segments ({:.0}%){}",
            completed, total, percentage, remaining_text
        ));
    }

    fn handle_completion(&self, result: Result<PathBuf, String>) {
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
    }

    fn handle_cancel(&self) {
        let mut ui = self.ui.borrow_mut();
        ui.is_translating.store(false, Ordering::SeqCst);
        ui.btn_cancel.set_hidden(true);
        ui.progress_bar.set_hidden(true);
        ui.status_label.set_text("Translation cancelled.");
        ui.translation_started_at = None;
    }
}

impl AppDelegate for TranslatorApp {
    fn did_finish_launching(&self) {
        App::set_menu(app_menus());
        App::activate();

        std::thread::spawn(|| {
            init_model();
        });

        self.window.set_minimum_content_size(420., 620.);
        self.window.set_title("Gemma EPUB Translator");
        self.window.set_content_view(&self.content_view);
        self.window.show();
    }

    fn open_urls(&self, urls: Vec<Url>) {
        if let Some(path) = first_epub_path_from_urls(&urls) {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::OpenFile(path));
        }
    }
}

impl Dispatcher for TranslatorApp {
    type Message = AppMessage;

    fn on_ui_message(&self, message: Self::Message) {
        match message {
            AppMessage::ShowOpenPanel => self.present_open_panel(),
            AppMessage::OpenFile(path) => self.handle_open_file(path),
            AppMessage::TranslationProgress { completed, total } => {
                self.handle_progress(completed, total);
            }
            AppMessage::TranslationComplete(result) => self.handle_completion(result),
            AppMessage::CancelTranslation => self.handle_cancel(),
        }
    }
}

fn app_menus() -> Vec<Menu> {
    vec![
        Menu::new(
            "",
            vec![
                MenuItem::Services,
                MenuItem::Separator,
                MenuItem::Hide,
                MenuItem::HideOthers,
                MenuItem::ShowAll,
                MenuItem::Separator,
                MenuItem::Quit,
            ],
        ),
        Menu::new(
            "File",
            vec![
                MenuItem::new("Open EPUB...")
                    .key("o")
                    .modifiers(&[EventModifierFlag::Command])
                    .action(|| {
                        App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ShowOpenPanel);
                    }),
                MenuItem::Separator,
                MenuItem::CloseWindow,
            ],
        ),
        Menu::new(
            "Edit",
            vec![
                MenuItem::Undo,
                MenuItem::Redo,
                MenuItem::Separator,
                MenuItem::Cut,
                MenuItem::Copy,
                MenuItem::Paste,
                MenuItem::Separator,
                MenuItem::SelectAll,
            ],
        ),
        Menu::new("View", vec![MenuItem::EnterFullScreen]),
        Menu::new(
            "Window",
            vec![
                MenuItem::Minimize,
                MenuItem::Zoom,
                MenuItem::Separator,
                MenuItem::new("Bring All to Front"),
            ],
        ),
    ]
}

fn is_epub_path(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("epub"))
        .unwrap_or(false)
}

fn first_epub_path_from_urls(urls: &[Url]) -> Option<PathBuf> {
    urls.iter()
        .filter_map(|url| url.to_file_path().ok())
        .find(is_epub_path)
}

fn first_epub_path_from_nsurls(urls: &[NSURL]) -> Option<PathBuf> {
    urls.iter().map(|url| url.pathbuf()).find(is_epub_path)
}
