use core::{
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
    sync::atomic::AtomicPtr,
};

use crate::track;

/// An allocated block before delivered to the user. That is to say, it contains
/// a valid block, which has a `next` pointer.
///
/// The block owns its underlying memory, although the corresponding size is
/// specified by its shard.
#[repr(transparent)]
#[must_use = "blocks must be used"]
pub(crate) struct BlockRef<'a>(NonNull<()>, PhantomData<&'a ()>);

// SAFETY: The block owns its underlying memory.
unsafe impl<'a> Send for BlockRef<'a> {}
unsafe impl<'a> Sync for BlockRef<'a> {}

impl<'a> BlockRef<'a> {
    const SLOT_SIZE: usize = mem::size_of::<Option<NonNull<()>>>();

    #[must_use = "blocks must be used"]
    pub(crate) fn into_raw(self) -> NonNull<()> {
        self.0
    }

    pub(crate) fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// # Safety
    ///
    /// The pointer must contain a valid block data.
    pub(crate) unsafe fn from_raw(ptr: NonNull<()>) -> Self {
        BlockRef(ptr, PhantomData)
    }

    /// # Safety
    ///
    /// The pointer must not be owned by other blocks.
    pub(super) unsafe fn new(ptr: NonNull<()>) -> Self {
        unsafe { Self::from_raw(ptr) }
    }

    pub fn set_next(&mut self, next: Option<Self>) {
        track::undefined(self.0.cast(), Self::SLOT_SIZE);
        // SAFETY: this structure contains a valid `next` pointer.
        unsafe { self.0.cast().write(next.map(Self::into_raw)) };
        track::no_access(self.0.cast(), Self::SLOT_SIZE);
    }

    pub fn take_next(&mut self) -> Option<Self> {
        let ptr = self.0.cast::<Option<NonNull<()>>>();
        track::defined(self.0.cast(), Self::SLOT_SIZE);
        // SAFETY: this structure contains a valid `next` pointer.
        let next = unsafe { ptr.read() };
        unsafe { ptr.write(None) };
        track::no_access(self.0.cast(), Self::SLOT_SIZE);
        next.map(|ptr| unsafe { Self::from_raw(ptr) })
    }

    pub fn set_tail(&mut self, data: Option<Self>) -> usize {
        let mut count = 1;
        let mut ptr = self.0.cast();
        loop {
            track::defined(ptr.cast(), Self::SLOT_SIZE);
            // SAFETY: this structure contains a valid `next` pointer.
            let next: Option<NonNull<()>> = unsafe { ptr.read() };

            match next {
                Some(next) => ptr = next.cast(),
                None => {
                    // SAFETY: this structure contains a valid `next` pointer.
                    unsafe { ptr.write(data.map(Self::into_raw)) };
                    track::no_access(ptr.cast(), Self::SLOT_SIZE);
                    break;
                }
            }
            count += 1;
            track::no_access(ptr.cast(), Self::SLOT_SIZE);
        }
        count
    }
}

impl Drop for BlockRef<'_> {
    fn drop(&mut self) {
        // We only create a blank implementation to mock its unique ownership,
        // and to prevent clippy from shouting `forget_non_drop`.
    }
}

/// An atomic slot containing an `Option<Block<'a>>`.
#[derive(Default)]
#[repr(transparent)]
pub(crate) struct AtomicBlockRef<'a>(AtomicPtr<()>, PhantomData<&'a ()>);

impl<'a> AtomicBlockRef<'a> {
    pub(crate) const fn new() -> Self {
        AtomicBlockRef(AtomicPtr::new(ptr::null_mut()), PhantomData)
    }

    pub(crate) fn get(&self) -> &AtomicPtr<()> {
        &self.0
    }
}
