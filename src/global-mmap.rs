mod thread;

use core::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    ptr::{self, NonNull},
};

pub type Base = crate::base::MmapAlloc;

pub type Heap<'a> = crate::heap::Heap<'a, Base>;
pub type Context<'a> = crate::heap::Context<'a, Base>;
pub type Arenas = crate::arena::Arenas<Base>;
pub type Error = crate::arena::Error<Base>;

pub static ARENAS: Arenas = Arenas::new(Base::new());

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Ferroc;

impl Ferroc {
    pub fn collect(&self, force: bool) {
        thread::with(|heap| heap.collect(force))
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, Error> {
        thread::with(|heap| heap.allocate(layout))
    }

    /// # Safety
    ///
    /// `ptr` must point to an owned, valid memory block of `layout`, previously
    /// allocated by a certain instance of `Heap` alive in the scope, created
    /// from the same arena.
    pub unsafe fn layout_of(&self, ptr: NonNull<u8>) -> Option<Layout> {
        thread::with(|heap| heap.layout_of(ptr))
    }

    /// # Safety
    ///
    /// See [`Allocator::deallocate`] for more information.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        thread::with(|heap| heap.deallocate(ptr, layout))
    }

    /// # Safety
    ///
    /// `ptr` must point to an owned, valid memory block, previously allocated
    /// by a certain instance of `Heap` alive in the scope.
    #[cfg(feature = "c")]
    #[inline]
    pub(crate) unsafe fn free(&self, ptr: NonNull<u8>) {
        thread::with(|heap| heap.free(ptr))
    }
}

unsafe impl Allocator for Ferroc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        thread::with(|heap| Allocator::allocate(&heap, layout))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        thread::with(|heap| Allocator::deallocate(&heap, ptr, layout))
    }
}

unsafe impl GlobalAlloc for Ferroc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocate(layout)
            .map_or(ptr::null_mut(), |ptr| ptr.as_ptr().cast())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(ptr) = NonNull::new(ptr) {
            self.deallocate(ptr, layout)
        }
    }
}
