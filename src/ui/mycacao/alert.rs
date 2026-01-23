use cacao::{
    foundation::{NSInteger, NSString},
    objc::{
        class, msg_send, msg_send_id,
        rc::{Id, Owned},
        runtime::Object,
    },
};

#[derive(Debug)]
pub struct AlertResult {
    pub confirmed: bool,
}

#[derive(Debug)]
pub struct Alert(Id<Object, Owned>);

impl Alert {
    /// Creates a basic `NSAlert`, storing a pointer to it in the Objective C runtime.
    /// You can show this alert by calling `show()`.
    pub fn new(title: &str, message: &str) -> Self {
        let title = NSString::new(title);
        let message = NSString::new(message);

        Alert(unsafe {
            let mut alert = msg_send_id![class!(NSAlert), new];
            let _: () = msg_send![&mut alert, setMessageText: &*title];
            let _: () = msg_send![&mut alert, setInformativeText: &*message];
            alert
        })
    }

    /// Shows this alert as a modal.
    pub fn show(&self) {
        unsafe {
            let _: () = msg_send![&*self.0, runModal];
        }
    }

    pub fn show_with_cancel(&self, confirm_label: &str, cancel_label: &str) -> AlertResult {
        let confirm_label = NSString::new(confirm_label);
        let cancel_label = NSString::new(cancel_label);

        unsafe {
            let _: () = msg_send![&*self.0, addButtonWithTitle: &*confirm_label];
            let _: () = msg_send![&*self.0, addButtonWithTitle: &*cancel_label];
            let response: NSInteger = msg_send![&*self.0, runModal];
            AlertResult {
                confirmed: response == 1000,
            }
        }
    }
}
