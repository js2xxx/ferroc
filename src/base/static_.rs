use core::{
    alloc::{AllocError, Layout},
    cell::UnsafeCell,
    mem,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, Ordering::*},
};

use super::{BaseAlloc, Chunk, StaticHandle};

/// An allocator managing a constant sized block of static memory.
///
/// By using a [`Static`] allocator, users can
/// [`manage`](crate::arena::Arenas::manage) external static memory other than
/// allocating from this allocator alone. However, `HEADER_CAP` must
/// be set carefully because arena headers need to be allocated from this
/// structure. As a reminder, 1 bit in an arena header manages a memory block of
/// `SLAB_SIZE`.
///
/// # Examples
///
/// ```rust,ignore
/// use ferroc::{
///     arena::Arenas,
///     base::{Static, Chunk},
///     heap::{Context, Heap},
/// };
///
/// static BASE: Static<100> = Static::new();
/// static ARENAS: Arenas<&Static<100>> = Arenas::new(&BASE);
///
/// // Add external static memory blocks.
/// ARENAS.manage(unsafe { Chunk::from_static(/* .. */) });
///
/// // Use the heap.
/// let cx = Context::new(&ARENAS);
/// let heap = Heap::new(&cx);
/// ```
#[derive(Debug)]
pub struct Static<const HEADER_CAP: usize> {
    memory: UnsafeCell<[usize; HEADER_CAP]>,
    top: AtomicPtr<()>,
}

impl<const HEADER_CAP: usize> Default for Static<HEADER_CAP> {
    fn default() -> Self {
        Self::INIT
    }
}

// `memory` is guarded by `top`.
unsafe impl<const HEADER_CAP: usize> Sync for Static<HEADER_CAP> {}

impl<const HEADER_CAP: usize> Static<HEADER_CAP> {
    pub const INIT: Self = Self::new();

    pub const fn new() -> Self {
        Static {
            memory: UnsafeCell::new([0; HEADER_CAP]),
            top: AtomicPtr::<()>::new(ptr::null_mut()),
        }
    }

    fn alloc_inner(&'static self, layout: Layout) -> Option<Chunk<&Self>> {
        let layout = layout.align_to(mem::align_of::<usize>()).ok()?;
        let (Ok(mut top) | Err(mut top)) =
            self.top
                .compare_exchange(ptr::null_mut(), self.memory.get().cast(), AcqRel, Acquire);
        loop {
            let aligned = (top.addr().checked_add(layout.align() - 1))? & !(layout.align() - 1);
            let end = aligned
                .checked_add(layout.size())
                .filter(|&end| end < self.memory.get().addr() + HEADER_CAP)?;
            let new = NonNull::new(top.with_addr(aligned))?;
            match self
                .top
                .compare_exchange_weak(top, top.with_addr(end), AcqRel, Acquire)
            {
                // SAFETY: The returned pointer points to an owned memory block of `layout` within
                // the range of `self.memory`.
                Ok(_) => break Some(unsafe { Chunk::from_static(new.cast(), layout) }),
                Err(t) => top = t,
            }
        }
    }
}

unsafe impl<const HEADER_CAP: usize> BaseAlloc for &'static Static<HEADER_CAP> {
    const IS_ZEROED: bool = true;

    type Handle = StaticHandle;

    type Error = AllocError;

    fn allocate(&self, layout: Layout, _commit: bool) -> Result<Chunk<Self>, AllocError> {
        self.alloc_inner(layout).ok_or(AllocError)
    }

    unsafe fn deallocate(_: &mut Chunk<Self>) {}
}
