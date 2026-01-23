use cacao::appkit::menu::{Menu, MenuItem};
use cacao::appkit::tabview::item::TabViewItem;
use cacao::appkit::tabview::traits::TabViewDelegate;
use cacao::appkit::tabview::TabView;
use cacao::appkit::window::Window;
use cacao::appkit::{App, AppDelegate};
use cacao::events::EventModifierFlag;
use cacao::foundation::id;
use cacao::layout::{Layout, LayoutConstraint};
use cacao::notification_center::Dispatcher;
use cacao::objc::msg_send;
use cacao::url::Url;
use cacao::view::View;
use std::path::PathBuf;

use crate::translate::init_model;
use crate::ui::epub_view::EpubView;

#[derive(Debug)]
pub enum AppMessage {
    ShowOpenPanel,
    OpenFile(PathBuf),
    TranslationProgress { completed: usize, total: usize },
    TranslationComplete(Result<PathBuf, String>),
    CancelTranslation,
}

pub struct TranslatorApp {
    window: Window,
    content_view: View,
    tab_view: TabView,
    epub_view: EpubView,
}

impl TranslatorApp {
    pub fn new() -> Self {
        let content_view = View::new();
        let tab_view = TabView::new();
        let epub_view = EpubView::new();

        let app = Self {
            window: Window::default(),
            content_view,
            tab_view,
            epub_view,
        };

        app.tab_view.set_delegate(&app);
        app
    }

    fn setup_tabs(&self) {
        let mut epub_tab = TabViewItem::new("epub");
        epub_tab.set_label("EPUB");
        set_tab_view_item_view(&epub_tab, self.epub_view.view_id());

        self.tab_view.add_tab_view_item(epub_tab);

        self.tab_view.select_tab_view_item_with_identifier("epub");
    }
}

impl TabViewDelegate for TranslatorApp {
    const NAME: &'static str = "TranslatorAppTabViewDelegate";

    fn did_select_tab_view_item(&self, _item: &TabViewItem) {}
}

impl AppDelegate for TranslatorApp {
    fn did_finish_launching(&self) {
        App::set_menu(app_menus());
        App::activate();

        std::thread::spawn(|| {
            init_model();
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
        crate::translate::shutdown_model();
    }

    fn open_urls(&self, urls: Vec<Url>) {
        self.epub_view.open_urls(urls);
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
            AppMessage::ShowOpenPanel => self.epub_view.present_open_panel(),
            AppMessage::OpenFile(path) => self.epub_view.handle_open_file(path),
            AppMessage::TranslationProgress { completed, total } => {
                self.epub_view.handle_progress(completed, total);
            }
            AppMessage::TranslationComplete(result) => {
                self.epub_view.handle_completion(result);
            }
            AppMessage::CancelTranslation => self.epub_view.handle_cancel(),
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
