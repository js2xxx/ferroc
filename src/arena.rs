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
    base::{BaseAlloc, Chunk},
    slab::{Slab, SlabRef},
};

const BYTE_WIDTH: usize = u8::BITS as usize;

pub const SLAB_SHIFT: usize = 2 + 10 + 10;
pub const SLAB_SIZE: usize = 1 << SLAB_SHIFT;

pub const fn slab_layout(n: usize) -> Layout {
    match Layout::from_size_align(n << SLAB_SHIFT, SLAB_SIZE) {
        Ok(layout) => layout,
        Err(_) => panic!("invalid slab layout"),
    }
}

pub(crate) const SHARD_SHIFT: usize = 6 + 10;
pub(crate) const SHARD_SIZE: usize = 1 << SHARD_SHIFT;

pub(crate) const SHARD_COUNT: usize = SLAB_SIZE / SHARD_SIZE;

struct Arena<B: BaseAlloc> {
    arena_id: usize,
    chunk: Chunk<B>,
    header: Chunk<B>,
    is_exclusive: bool,
    slab_count: usize,
    search_index: AtomicUsize,
}

impl<B: BaseAlloc> Arena<B> {
    const LAYOUT: Layout = Layout::new::<Self>();

    fn header_layout(slab_count: usize, is_exclusive: bool) -> Layout {
        if !is_exclusive {
            let bitmap_size = (slab_count * SLAB_SIZE).div_ceil(BYTE_WIDTH);
            let n = bitmap_size.div_ceil(mem::size_of::<AtomicUsize>());

            let bitmap_layout = Layout::array::<AtomicUsize>(n).unwrap();
            let (header_layout, offset) = Self::LAYOUT.extend(bitmap_layout).unwrap();
            assert_eq!(offset, Self::LAYOUT.size());
            header_layout
        } else {
            Self::LAYOUT
        }
    }

    /// # Safety
    ///
    /// `header` must contains a valid bitmap covering exact all memory blocks
    /// in `chunk` (i.e. `slab_count`).
    unsafe fn from_dynamic<'a>(
        header: Chunk<B>,
        chunk: Chunk<B>,
        is_exclusive: bool,
        slab_count: usize,
    ) -> &'a mut Arena<B> {
        // SAFETY: The pointer is properly aligned.
        let arena = unsafe {
            let pointer = header.pointer().cast::<Arena<B>>();
            pointer.as_uninit_mut().write(Arena {
                arena_id: 0,
                chunk,
                header,
                is_exclusive,
                slab_count,
                search_index: Default::default(),
            })
        };

        if !is_exclusive {
            // SAFETY: the bitmap pointer points to a valid & uninit memory block.
            unsafe {
                let maybe = arena.bitmap_ptr().as_uninit_slice_mut();
                maybe.fill(MaybeUninit::new(0));
            }
            let bitmap = arena.bitmap();
            bitmap.set::<true>(slab_count.try_into().unwrap()..bitmap.len());
        }

        arena
    }

    fn new<'a>(
        base: B,
        slab_count: usize,
        align: Option<usize>,
        is_exclusive: bool,
    ) -> Result<&'a mut Self, Error<B>> {
        let layout = match align {
            Some(align) => slab_layout(slab_count)
                .align_to(align)
                .expect("invalid align"),
            None => slab_layout(slab_count),
        };

        let header_layout = Self::header_layout(slab_count, is_exclusive);

        let chunk = base.clone().allocate(layout).map_err(Error::Base)?;
        let header = base.allocate(header_layout).map_err(Error::Base)?;

        // SAFETY: `header` is valid.
        Ok(unsafe { Self::from_dynamic(header, chunk, is_exclusive, slab_count) })
    }

    fn new_chunk<'a>(base: B, chunk: Chunk<B>) -> Result<&'a mut Self, Error<B>> {
        let size = chunk.pointer().len();
        let slab_count = size / SLAB_SIZE;
        assert!(chunk.pointer().is_aligned_to(SLAB_SIZE));

        let header_layout = Self::header_layout(slab_count, false);
        let header = base.allocate(header_layout).map_err(Error::Base)?;

        Ok(unsafe { Self::from_dynamic(header, chunk, false, slab_count) })
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
        Some(unsafe { Slab::init(ptr, thread_id, self.arena_id, is_huge, B::IS_ZEROED) })
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

const MAX_ARENAS: usize = 112;
pub struct Arenas<B: BaseAlloc> {
    pub(crate) base: B,
    arenas: [AtomicPtr<Arena<B>>; MAX_ARENAS],
    arena_count: AtomicUsize,
    slab_count: AtomicUsize,
    abandoned: AtomicPtr<()>,
}

impl<B: BaseAlloc> Arenas<B> {
    // We're using this constant to initialize the array, so no real manipulation on
    // this constant is performed.
    #[allow(clippy::declare_interior_mutable_const)]
    const ARENA_INIT: AtomicPtr<Arena<B>> = AtomicPtr::new(ptr::null_mut());

    pub const fn new(base: B) -> Self {
        Arenas {
            base,
            arenas: [Self::ARENA_INIT; MAX_ARENAS],
            arena_count: AtomicUsize::new(0),
            slab_count: AtomicUsize::new(0),
            abandoned: AtomicPtr::new(ptr::null_mut()),
        }
    }

    fn push_arena<'a>(&'a self, arena: &'a mut Arena<B>) -> Result<&'a Arena<B>, Error<B>> {
        let index = self.arena_count.fetch_add(1, AcqRel);
        if index < MAX_ARENAS {
            arena.arena_id = index + 1;
            self.arenas[index].store(arena, Release);
            Ok(arena)
        } else if let Some(index) = self.arenas.iter().position(|slot| {
            slot.compare_exchange(ptr::null_mut(), arena, AcqRel, Acquire)
                .is_ok()
        }) {
            arena.arena_id = index + 1;
            Ok(arena)
        } else {
            self.arena_count.fetch_sub(1, AcqRel);
            // SAFETY: The arena is freshly allocated.
            unsafe { Arena::drop(arena.into()) };
            Err(Error::ArenaExhausted)
        }
    }

    fn push_abandoned(&self, slab: SlabRef) {
        assert!(slab.is_abandoned());
        let mut next = self.abandoned.load(Relaxed);
        loop {
            slab.abandoned_next.store(next, Relaxed);
            match self
                .abandoned
                .compare_exchange(next, slab.as_ptr().as_ptr(), AcqRel, Acquire)
            {
                Ok(_) => break,
                Err(e) => next = e,
            }
        }
    }

    pub(crate) fn collect_abandoned(&self) {
        let mut next = self.abandoned.swap(ptr::null_mut(), Relaxed);
        // SAFETY: `pointer` is owned: abandoned slabs comes from a dead thread context.
        while let Some(slab) = NonNull::new(next).map(|p| unsafe { SlabRef::from_ptr(p) }) {
            next = slab.abandoned_next.load(Relaxed);
            slab.collect_abandoned();
            unsafe { self.deallocate(slab) };
        }
    }

    fn arenas(&self, is_exclusive: bool) -> impl Iterator<Item = (usize, &Arena<B>)> {
        let iter = self.arenas[..self.arena_count.load(Acquire)].iter();
        // SAFETY: We check the nullity of the pointers.
        iter.enumerate()
            .filter_map(|(index, arena)| Some((index, unsafe { arena.load(Acquire).as_ref() }?)))
            .filter(move |(_, arena)| arena.is_exclusive == is_exclusive)
    }

    pub fn os_alloc(&self) -> &B {
        &self.base
    }

    pub fn manage(&self, chunk: Chunk<B>) -> Result<(), Error<B>> {
        let arena = Arena::new_chunk(self.base.clone(), chunk)?;
        self.push_arena(arena)?;
        Ok(())
    }

    pub(crate) fn allocate(
        &self,
        thread_id: u64,
        count: NonZeroUsize,
        align: usize,
        is_huge: bool,
    ) -> Result<SlabRef, Error<B>> {
        let count = count.get().max(self.slab_count.load(Relaxed).isqrt());

        self.collect_abandoned();
        let ret = match self
            .arenas(false)
            .find_map(|(_, arena)| arena.allocate(thread_id, count, align, is_huge))
        {
            Some(slab) => slab,
            None => {
                let arena = Arena::new(self.base.clone(), count, Some(align), false)?;
                let arena = self.push_arena(arena)?;
                arena.allocate(thread_id, count, align, is_huge).unwrap()
            }
        };
        self.slab_count.fetch_add(count, Relaxed);
        Ok(ret)
    }

    /// # Safety
    ///
    /// `slab` must be previously allocated from this structure;
    pub(crate) unsafe fn deallocate(&self, slab: SlabRef) {
        if !slab.is_abandoned() {
            let arena = self.arenas[slab.arena_id - 1].load(Acquire);
            debug_assert!(!arena.is_null());
            // SAFETY: `arena` is obtained from the unique `arena_id`, and the arena won't
            // be dropped as long as any allocation from it is alive.
            let slab_count = unsafe { (*arena).deallocate(slab) };
            self.slab_count.fetch_sub(slab_count, Relaxed);
        } else {
            self.push_abandoned(slab)
        }
    }

    pub fn allocate_direct(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<B>> {
        let arena = Arena::new(
            self.base.clone(),
            layout.size().div_ceil(SLAB_SIZE),
            Some(layout.align()),
            true,
        )?;
        let arena = self.push_arena(arena)?;
        Ok(NonNull::from_raw_parts(
            arena.chunk.pointer().cast(),
            layout.size(),
        ))
    }

    /// # Panics
    ///
    /// Panics if `ptr` is not allocated from this structure.
    ///
    /// # Safety
    ///
    /// No more aliases to the `ptr` should exist after calling this function.
    pub unsafe fn deallocate_direct(&self, ptr: NonNull<u8>) {
        if let Some((index, _)) = self
            .arenas(true)
            .find(|(_, arena)| arena.chunk.pointer().cast() == ptr)
        {
            let arena = self.arenas[index].swap(ptr::null_mut(), AcqRel);
            debug_assert!(!arena.is_null());
            // SAFETY: `arena` is exclusive.
            unsafe { Arena::drop(NonNull::new_unchecked(arena)) }
        } else {
            #[cfg(debug_assertions)]
            panic!("deallocating memory not from these arenas")
        }
    }
}

impl<B: BaseAlloc> Drop for Arenas<B> {
    fn drop(&mut self) {
        let iter = self.arenas[..*self.arena_count.get_mut()].iter_mut();
        iter.filter_map(|arena| NonNull::new(mem::replace(arena.get_mut(), ptr::null_mut())))
            // SAFETY: All the arenas are unreferenced due to the lifetime model.
            .for_each(|arena| unsafe { Arena::drop(arena) })
    }
}

#[derive(Debug)]
pub enum Error<B: BaseAlloc> {
    Base(B::Error),
    ArenaExhausted,
}
