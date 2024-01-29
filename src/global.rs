use core::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    mem,
    ptr::{self, NonNull},
};

#[cfg(feature = "os-mmap")]
pub type Os = crate::os::mmap::MmapAlloc;

pub type Heap<'a> = crate::heap::Heap<'a, Os>;
pub type Context<'a> = crate::heap::Context<'a, Os>;
pub type Arenas = crate::arena::Arenas<Os>;
pub type Error = crate::arena::Error<Os>;

pub static ARENAS: Arenas = Arenas::new(Os::new());

thread_local! {
    static CX: Context<'static> = Context::new(&ARENAS);

    // SAFETY: The safety transmutation relies on the behavior of "first construct,
    // last destruct".
    static HEAP: Heap<'static> = Heap::new(CX.with(|cx| unsafe { mem::transmute(cx) }));
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Ferroc;

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
