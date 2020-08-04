/// A Container type
///
/// Provides an interface to allow transfer of ownership between Node and Rust.
/// It either contains a Box with full ownership of the content, or a pointer to the content.
///
/// The main goal here is to allow Node calling into Rust to initialize some objects. Later
/// these objects may need to be used by Rust who will expect to take ownership. Since Node
/// does not allow any sort of ownership transfer, it will keep a reference to this object
/// until it gets cleaned up by the GC. In this case, we actually give the ownership to Rust,
/// and just keep a pointer in the Node object.
pub enum Container<T: ?Sized> {
    Owned(Box<T>),
    Empty,
}

impl<T> Container<T>
where
    T: ?Sized,
{
    /// Replace an empty content by the new provided owned one, otherwise do nothing
    pub fn make_owned(&mut self, o: Box<T>) {
        if let Container::Empty = self {
            unsafe {
                let new_container = Container::Owned(o);
                std::ptr::write(self, new_container);
            }
        }
    }

    pub fn execute<F, U>(&self, closure: F) -> U
    where
        F: FnOnce(Option<&Box<T>>) -> U,
    {
        match self {
            Container::Owned(val) => closure(Some(val)),
            Container::Empty => closure(None),
        }
    }

    pub fn execute_mut<F, U>(&mut self, closure: F) -> U
    where
        F: FnOnce(Option<&mut Box<T>>) -> U,
    {
        match self {
            Container::Owned(val) => closure(Some(val)),
            Container::Empty => closure(None),
        }
    }
}
