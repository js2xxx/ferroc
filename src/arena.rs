mod bitmap;

use core::{
    alloc::Layout,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    panic,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering::*},
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
    arena_id: usize,
    chunk: Chunk<Os>,
    header: Chunk<Os>,
    slab_count: usize,
    search_index: AtomicUsize,
}

impl<Os: OsAlloc> Arena<Os> {
    const LAYOUT: Layout = Layout::new::<Self>();

    pub fn new<'a>(
        os: Os,
        slab_count: usize,
        align: Option<usize>,
    ) -> Result<&'a mut Self, Error<Os>> {
        let layout = match align {
            Some(align) => slab_layout(slab_count)
                .align_to(align)
                .expect("invalid align"),
            None => slab_layout(slab_count),
        };

        let bitmap_size = (slab_count * SLAB_SIZE).div_ceil(BYTE_WIDTH);
        let n = bitmap_size.div_ceil(mem::size_of::<AtomicUsize>());

        let bitmap_layout = Layout::array::<AtomicUsize>(n).unwrap();
        let (header_layout, offset) = Self::LAYOUT.extend(bitmap_layout).unwrap();
        assert_eq!(offset, Self::LAYOUT.size());

        let chunk = os.clone().allocate(layout).map_err(Error::Os)?;
        let header = os.allocate(header_layout).map_err(Error::Os)?;

        // SAFETY: The pointer is properly aligned.
        let arena = unsafe {
            let pointer = header.pointer().cast::<Self>();
            pointer.as_uninit_mut().write(Arena {
                arena_id: 0,
                chunk,
                header,
                slab_count,
                search_index: Default::default(),
            })
        };

        // SAFETY: the bitmap pointer points to a valid & uninit memory block.
        unsafe {
            let maybe = arena.bitmap_ptr().as_uninit_slice_mut();
            maybe.fill(MaybeUninit::new(0));
        }
        let bitmap = arena.bitmap();
        bitmap.set::<true>(slab_count.try_into().unwrap()..bitmap.len());

        Ok(arena)
    }

    /// # Safety
    ///
    /// `arena` must have no other references alive.
    unsafe fn drop(arena: NonNull<Self>) {
        // SAFETY: We read the data first so as to avoid dropping the dropped data.
        drop(unsafe { arena.read() });
    }

    fn bitmap_ptr(&self) -> NonNull<[u8]> {
        let (ptr, _) = self.header.pointer().to_raw_parts();
        NonNull::from_raw_parts(
            ptr.map_addr(|addr| addr.checked_add(Self::LAYOUT.size()).unwrap()),
            (self.slab_count * SLAB_SIZE).div_ceil(BYTE_WIDTH),
        )
    }

    fn bitmap(&self) -> &Bitmap {
        let (ptr, len) = self.bitmap_ptr().to_raw_parts();
        let slice = NonNull::from_raw_parts(ptr, len / mem::size_of::<AtomicUsize>());
        // SAFETY: The bitmap pointer points to a valid `[AtomicUsize]`.
        Bitmap::new(unsafe { slice.as_ref() })
    }

    fn allocate_slices(&self, count: usize) -> Option<NonNull<[u8]>> {
        let start = self.search_index.load(Relaxed);
        let (idx, bit) = self.bitmap().allocate(start, count.try_into().ok()?)?;
        self.search_index.store(idx, Relaxed);

        let offset = (idx * BYTE_WIDTH + (bit as usize)) * SLAB_SIZE;
        Some(NonNull::slice_from_raw_parts(
            // SAFETY: `idx` and `bit` is valid, and thus `offset` is within the chunk memory
            // range.
            unsafe { self.chunk.pointer().cast().add(offset) },
            SLAB_SIZE * count,
        ))
    }

    fn allocate(
        &self,
        thread_id: u64,
        count: usize,
        align: usize,
        is_huge: bool,
    ) -> Option<SlabRef> {
        debug_assert!(align <= SLAB_SIZE);
        let ptr = self.allocate_slices(count)?;
        // SAFETY: The fresh allocation is aligned to `SLAB_SIZE`.
        Some(unsafe { Slab::init(ptr, thread_id, self.arena_id, is_huge, Os::IS_ZEROED) })
    }

    /// # Safety
    ///
    /// - `slab` must be previously allocated from this arena;
    /// - No more references to the `slab` or its shards exist after calling
    ///   this function.
    unsafe fn deallocate(&self, slab: SlabRef) -> usize {
        let (ptr, len) = slab.into_raw().to_raw_parts();
        let offset = unsafe { ptr.cast::<u8>().sub_ptr(self.chunk.pointer().cast()) };

        let (start, end) = (offset / SLAB_SIZE, (offset + len) / SLAB_SIZE);
        self.bitmap().set::<false>((start as u32)..(end as u32));
        end - start
    }
}

const MAX_ARENAS: usize = 32;
pub struct Arenas<Os: OsAlloc> {
    pub(crate) os: Os,
    arenas: [AtomicPtr<Arena<Os>>; MAX_ARENAS],
    arena_count: AtomicUsize,
    slab_count: AtomicUsize,
}

impl<Os: OsAlloc> Arenas<Os> {
    // We're using this constant to initialize the array, so no real manipulation on
    // this constant is performed.
    #[allow(clippy::declare_interior_mutable_const)]
    const ARENA_INIT: AtomicPtr<Arena<Os>> = AtomicPtr::new(ptr::null_mut());

    pub fn new(os: Os) -> Self {
        Arenas {
            os,
            arenas: [Self::ARENA_INIT; MAX_ARENAS],
            arena_count: AtomicUsize::new(0),
            slab_count: AtomicUsize::new(0),
        }
    }

    fn push_arena(
        &self,
        slab_count: usize,
        align: Option<usize>,
    ) -> Result<Option<&Arena<Os>>, Error<Os>> {
        let arena = Arena::new(self.os.clone(), slab_count, align)?;
        let index = self.arena_count.fetch_add(1, AcqRel);
        Ok(if index >= MAX_ARENAS {
            // SAFETY: The arena is freshly allocated.
            unsafe { Arena::drop(arena.into()) };
            None
        } else {
            arena.arena_id = index + 1;
            self.arenas[index].store(arena, Release);
            Some(arena)
        })
    }

    fn arenas(&self) -> impl Iterator<Item = &Arena<Os>> {
        let iter = self.arenas[..self.arena_count.load(Acquire)].iter();
        // SAFETY: We check the nullity of the pointers.
        iter.filter_map(|arena| unsafe { arena.load(Acquire).as_ref() })
    }

    pub fn allocate(
        &self,
        thread_id: u64,
        count: NonZeroUsize,
        align: usize,
        is_huge: bool,
    ) -> Result<SlabRef, Error<Os>> {
        let count = count.get().max(self.slab_count.load(Relaxed).isqrt());
        let ret = match self
            .arenas()
            .find_map(|arena| arena.allocate(thread_id, count, align, is_huge))
        {
            Some(slab) => slab,
            None => {
                let arena = self.push_arena(count, Some(align))?;
                let arena = arena.ok_or(Error::ArenaExhausted)?;
                arena.allocate(thread_id, count, align, is_huge).unwrap()
            }
        };
        self.slab_count.fetch_add(count, Relaxed);
        Ok(ret)
    }

    /// # Safety
    ///
    /// - `slab` must be previously allocated from this structure;
    /// - No more references to the `slab` or its shards exist after calling
    ///   this function.
    pub unsafe fn deallocate(&self, slab: SlabRef) {
        let arena = self.arenas[slab.arena_id - 1].load(Acquire);
        debug_assert!(!arena.is_null());
        // SAFETY: `arena` is obtained from the unique `arena_id`, and the arena won't
        // be dropped as long as any allocation from it is alive.
        let slab_count = unsafe { (*arena).deallocate(slab) };
        self.slab_count.fetch_sub(slab_count, Relaxed);
    }
}

impl<Os: OsAlloc> Drop for Arenas<Os> {
    fn drop(&mut self) {
        let iter = self.arenas[..*self.arena_count.get_mut()].iter_mut();
        iter.filter_map(|arena| NonNull::new(mem::replace(arena.get_mut(), ptr::null_mut())))
            // SAFETY: All the arenas are unreferenced due to the lifetime model.
            .for_each(|arena| unsafe { Arena::drop(arena) })
    }
}

#[derive(Debug)]
pub enum Error<Os: OsAlloc> {
    Os(Os::Error),
    ArenaExhausted,
}
