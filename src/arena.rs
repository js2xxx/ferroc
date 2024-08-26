mod bitmap;

use core::{
    alloc::Layout,
    mem::{self, MaybeUninit},
    panic,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use self::bitmap::Bitmap;
use crate::{
    os::{Chunk, OsAlloc},
    slab::{Slab, SlabRef},
};

const BYTE_WIDTH: usize = u8::BITS as usize;

pub(crate) const SLAB_SHIFT: usize = 2 + 10 + 10;
pub(crate) const SLAB_SIZE: usize = 1 << SLAB_SHIFT;

pub const fn slab_layout(n: usize) -> Layout {
    match Layout::from_size_align(n << SLAB_SHIFT, SLAB_SIZE) {
        Ok(layout) => layout,
        Err(_) => panic!("invalid slab layout"),
    }
}

pub(crate) const SHARD_SHIFT: usize = 6 + 10;
pub(crate) const SHARD_SIZE: usize = 1 << SHARD_SHIFT;

pub(crate) const SHARD_COUNT: usize = SLAB_SIZE / SHARD_SIZE;

pub struct Arena<Os: OsAlloc> {
    chunk: Chunk<Os>,
    // slab_count: usize,
    header_count: usize,

    search_index: AtomicUsize,
}

impl<Os: OsAlloc> Arena<Os> {
    pub fn new(chunk: Chunk<Os>) -> Self {
        assert!(chunk.layout().size() >= mem::size_of::<AtomicUsize>());
        assert!(chunk.layout().align() >= mem::align_of::<AtomicUsize>());
        let layout = chunk.layout();
        let slab_count = layout.size().div_ceil(SLAB_SIZE);

        let bitmap_size = slab_count.div_ceil(BYTE_WIDTH);
        let bitmap_count = bitmap_size.div_ceil(SLAB_SIZE);

        let header_count = bitmap_count;

        let arena = Arena {
            chunk,
            // slab_count,
            header_count,
            search_index: Default::default(),
        };

        // SAFETY: the bitmap pointer points to a valid & uninit memory block.
        unsafe {
            let maybe = arena.bitmap_ptr().as_uninit_slice_mut();
            maybe.fill(MaybeUninit::new(0));
        }
        let bitmap = arena.bitmap();
        bitmap.set::<true>(0..header_count.try_into().unwrap());
        bitmap.set::<true>(slab_count.try_into().unwrap()..bitmap.len());

        arena
    }

    fn bitmap_ptr(&self) -> NonNull<[u8]> {
        let pointer = self.chunk.pointer();
        let size = self.header_count * SLAB_SIZE;
        NonNull::slice_from_raw_parts(pointer.as_non_null_ptr(), size)
    }

    fn bitmap(&self) -> &Bitmap {
        let (ptr, len) = self.bitmap_ptr().to_raw_parts();
        let slice = NonNull::from_raw_parts(ptr, len / mem::size_of::<AtomicUsize>());
        // SAFETY: The bitmap pointer points to a valid `[AtomicUsize]`.
        Bitmap::new(unsafe { slice.as_ref() })
    }

    pub fn allocate_slices(&self, count: usize) -> Option<NonNull<[u8]>> {
        let start = self.search_index.load(Relaxed);
        let (idx, bit) = self.bitmap().allocate(start, count.try_into().ok()?)?;
        self.search_index.store(idx, Relaxed);

        let offset = (idx * BYTE_WIDTH + (bit as usize)) * SLAB_SIZE;
        Some(NonNull::slice_from_raw_parts(
            unsafe { self.chunk.pointer().cast().add(offset) },
            SLAB_SIZE * count,
        ))
    }

    pub fn allocate(&self, id: u64) -> Option<SlabRef> {
        // SAFETY: The fresh allocation is aligned to `SLAB_SIZE`.
        Some(unsafe { Slab::init(self.allocate_slices(1)?, id, Os::IS_ZEROED) })
    }

    /// # Safety
    ///
    /// - `slab` must be previously allocated from this arena;
    /// - No more references to the `slab` or its shards exist after calling
    ///   this function.
    pub unsafe fn deallocate(&self, slab: SlabRef) {
        let (ptr, len) = slab.into_raw().to_raw_parts();
        let offset = unsafe { ptr.cast::<u8>().sub_ptr(self.chunk.pointer().cast()) };

        let (start, end) = (offset / SLAB_SIZE, (offset + len) / SLAB_SIZE);
        self.bitmap().set::<false>((start as u32)..(end as u32));
    }
}
