use core::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    mem,
    ptr::{self, NonNull},
};

#[cfg(feature = "base-mmap")]
pub type Base = crate::base::MmapAlloc;

pub type Heap<'a> = crate::heap::Heap<'a, Base>;
pub type Context<'a> = crate::heap::Context<'a, Base>;
pub type Arenas = crate::arena::Arenas<Base>;
pub type Error = crate::arena::Error<Base>;

pub static ARENAS: Arenas = Arenas::new(Base::new());

thread_local! {
    static CX: Context<'static> = Context::new(&ARENAS);

    // SAFETY: The safety transmutation relies on the behavior of "first construct,
    // last destruct".
    static HEAP: Heap<'static> = Heap::new(CX.with(|cx| unsafe { mem::transmute(cx) }));
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Ferroc;

impl Ferroc {
    pub fn collect(&self, force: bool) {
        HEAP.with(|heap| heap.collect(force))
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, Error> {
        HEAP.with(|heap| heap.allocate(layout))
    }

    /// # Safety
    ///
    /// See [`Allocator::deallocate`] for more information.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        HEAP.with(|heap| heap.deallocate(ptr, layout))
    }
}

unsafe impl Allocator for Ferroc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        HEAP.with(|heap| Allocator::allocate(&heap, layout))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        HEAP.with(|heap| Allocator::deallocate(&heap, ptr, layout))
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
