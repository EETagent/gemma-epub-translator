use cacao::appkit::menu::{Menu, MenuItem};
use cacao::appkit::tabview::item::TabViewItem;
use cacao::appkit::tabview::traits::TabViewDelegate;
use cacao::appkit::tabview::TabView;
use cacao::appkit::window::Window;
use cacao::appkit::window::{WindowConfig, WindowDelegate};
use cacao::appkit::{App, AppDelegate};
use cacao::events::EventModifierFlag;
use cacao::foundation::id;
use cacao::foundation::nil;
use cacao::layout::{Layout, LayoutConstraint};
use cacao::notification_center::Dispatcher;
use cacao::objc::msg_send;
use cacao::url::Url;
use cacao::view::View;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::translate::{create_state, LlamaState};
use crate::ui::mycacao::alert::Alert;
use crate::ui::views::epub_view::EpubView;
use crate::ui::views::srt_view::SrtView;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationKind {
    Epub,
    Srt,
}

#[derive(Debug)]
pub enum AppMessage {
    ShowOpenPanel(TranslationKind),
    OpenFile {
        kind: TranslationKind,
        path: PathBuf,
    },
    TranslationProgress {
        kind: TranslationKind,
        completed: usize,
        total: usize,
    },
    TranslationComplete {
        kind: TranslationKind,
        result: Result<PathBuf, String>,
    },
    CancelTranslation(TranslationKind),
    ModelLoaded,
    ModelLoadFailed(String),
}

pub struct TranslatorApp {
    window: Window<AppWindowDelegate>,
    content_view: View,
    tab_view: TabView,
    epub_view: EpubView,
    srt_view: SrtView,
    model_state: Arc<Mutex<Option<Arc<Mutex<LlamaState>>>>>,
}

struct AppWindowDelegate {
    is_translating_epub: Arc<AtomicBool>,
    is_translating_srt: Arc<AtomicBool>,
}

impl WindowDelegate for AppWindowDelegate {
    const NAME: &'static str = "TranslatorWindowDelegate";

    fn should_close(&self) -> bool {
        if !self.is_translating_epub.load(Ordering::SeqCst)
            && !self.is_translating_srt.load(Ordering::SeqCst)
        {
            return true;
        }

        let alert = Alert::new(
            "Translation in progress",
            "Closing the window will cancel the current translation and lose progress.",
        );
        let result = alert.show_with_cancel("Close", "Cancel");
        if result.confirmed {
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::CancelTranslation(
                TranslationKind::Epub,
            ));
            App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::CancelTranslation(
                TranslationKind::Srt,
            ));
            true
        } else {
            false
        }
    }
}

impl TranslatorApp {
    pub fn new() -> Self {
        let content_view = View::new();
        let tab_view = TabView::new();
        let epub_view = EpubView::new();
        let srt_view = SrtView::new();
        let window_delegate = AppWindowDelegate {
            is_translating_epub: epub_view.is_translating_flag(),
            is_translating_srt: srt_view.is_translating_flag(),
        };

        let app = Self {
            window: Window::with(WindowConfig::default(), window_delegate),
            content_view,
            tab_view,
            epub_view,
            srt_view,
            model_state: Arc::new(Mutex::new(None)),
        };

        app.tab_view.set_delegate(&app);
        app
    }

    fn setup_tabs(&self) {
        let mut epub_tab = TabViewItem::new("epub");
        epub_tab.set_label("EPUB");
        set_tab_view_item_view(&epub_tab, self.epub_view.view_id());

        let mut srt_tab = TabViewItem::new("srt");
        srt_tab.set_label("SRT");
        set_tab_view_item_view(&srt_tab, self.srt_view.view_id());

        self.tab_view.add_tab_view_item(epub_tab);
        self.tab_view.add_tab_view_item(srt_tab);

        self.tab_view.select_tab_view_item_with_identifier("epub");
    }
}

impl TabViewDelegate for TranslatorApp {
    const NAME: &'static str = "TranslatorAppTabViewDelegate";

    fn did_select_tab_view_item(&self, _item: &TabViewItem) {
    }
}

impl AppDelegate for TranslatorApp {
    fn should_terminate_after_last_window_closed(&self) -> bool {
        true
    }

    fn did_finish_launching(&self) {
        App::set_menu(app_menus());
        App::activate();

        self.epub_view.set_loading_model();
        self.srt_view.set_loading_model();

        let model_state_ref = self.model_state.clone();
        std::thread::spawn(move || match create_state() {
            Ok(state) => {
                let state = Arc::new(Mutex::new(state));
                *model_state_ref.lock().unwrap() = Some(state);
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ModelLoaded);
            }
            Err(err) => {
                App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ModelLoadFailed(
                    err.to_string(),
                ));
            }
        });

        self.window.set_minimum_content_size(520., 720.);
        self.window.set_title("Gemma EPUB Translator");
        self.window.set_content_view(&self.content_view);

        self.content_view.add_subview(&self.tab_view);

        self.setup_tabs();

        LayoutConstraint::activate(&[
            self.tab_view
                .top
                .constraint_equal_to(&self.content_view.safe_layout_guide.top),
            self.tab_view
                .leading
                .constraint_equal_to(&self.content_view.safe_layout_guide.leading),
            self.tab_view
                .trailing
                .constraint_equal_to(&self.content_view.safe_layout_guide.trailing),
            self.tab_view
                .bottom
                .constraint_equal_to(&self.content_view.safe_layout_guide.bottom),
        ]);

        self.window.show();
    }

    fn will_terminate(&self) {
        if self.epub_view.is_translating_flag().load(Ordering::SeqCst)
            || self.srt_view.is_translating_flag().load(Ordering::SeqCst)
        {
            self.epub_view.handle_cancel();
            self.srt_view.handle_cancel();
            std::thread::sleep(std::time::Duration::from_millis(750));
        }

        self.epub_view.clear_model_state();
        self.srt_view.clear_model_state();

        if let Ok(mut lock) = self.model_state.lock() {
            *lock = None;
        }

        //std::process::exit(0);
    }

    fn open_urls(&self, urls: Vec<Url>) {
        self.epub_view.open_urls(urls.clone());
        self.srt_view.open_urls(urls);
    }
}

fn set_tab_view_item_view(item: &TabViewItem, view: id) {
    unsafe {
        let _: () = msg_send![&*item.objc, setView: view];
    }
}

impl Dispatcher for TranslatorApp {
    type Message = AppMessage;

    fn on_ui_message(&self, message: Self::Message) {
        match message {
            AppMessage::ShowOpenPanel(kind) => match kind {
                TranslationKind::Epub => self.epub_view.present_open_panel(),
                TranslationKind::Srt => self.srt_view.present_open_panel(),
            },
            AppMessage::OpenFile { kind, path } => match kind {
                TranslationKind::Epub => self.epub_view.handle_open_file(path),
                TranslationKind::Srt => self.srt_view.handle_open_file(path),
            },
            AppMessage::TranslationProgress {
                kind,
                completed,
                total,
            } => match kind {
                TranslationKind::Epub => self.epub_view.handle_progress(completed, total),
                TranslationKind::Srt => self.srt_view.handle_progress(completed, total),
            },
            AppMessage::TranslationComplete { kind, result } => match kind {
                TranslationKind::Epub => self.epub_view.handle_completion(result),
                TranslationKind::Srt => self.srt_view.handle_completion(result),
            },
            AppMessage::CancelTranslation(kind) => match kind {
                TranslationKind::Epub => self.epub_view.handle_cancel(),
                TranslationKind::Srt => self.srt_view.handle_cancel(),
            },
            AppMessage::ModelLoaded => {
                if let Some(state) = self.model_state.lock().unwrap().clone() {
                    self.epub_view.set_model_state(state.clone());
                    self.srt_view.set_model_state(state);
                }
            }
            AppMessage::ModelLoadFailed(err) => {
                self.epub_view.set_model_load_error(&err);
                self.srt_view.set_model_load_error(&err);
            }
        }
    }
}

pub fn app_menus() -> Vec<Menu> {
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
                        App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ShowOpenPanel(
                            TranslationKind::Epub,
                        ));
                    }),
                MenuItem::new("Open SRT...")
                    .key("o")
                    .modifiers(&[EventModifierFlag::Command, EventModifierFlag::Shift])
                    .action(|| {
                        App::<TranslatorApp, AppMessage>::dispatch_main(AppMessage::ShowOpenPanel(
                            TranslationKind::Srt,
                        ));
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
