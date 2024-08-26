//! The module of the arena collection.
//!
//! See [`Arenas`] for information.

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
    track,
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
            let bitmap_size = slab_count
                .div_ceil(BYTE_WIDTH)
                .next_multiple_of(mem::size_of::<AtomicUsize>());
            let n = bitmap_size / mem::size_of::<AtomicUsize>();

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
        base: &B,
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

        let chunk = base.allocate(layout, is_exclusive).map_err(Error::Alloc)?;
        let header = base.allocate(header_layout, true).map_err(Error::Alloc)?;

        // SAFETY: `header` is valid.
        Ok(unsafe { Self::from_dynamic(header, chunk, is_exclusive, slab_count) })
    }

    fn new_chunk<'a>(base: &B, chunk: Chunk<B>) -> Result<&'a mut Self, Error<B>> {
        let size = chunk.pointer().len();
        let slab_count = size / SLAB_SIZE;
        assert!(chunk.pointer().is_aligned_to(SLAB_SIZE));

        let header_layout = Self::header_layout(slab_count, false);
        let header = base.allocate(header_layout, true).map_err(Error::Alloc)?;

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
            self.slab_count
                .div_ceil(BYTE_WIDTH)
                .next_multiple_of(mem::size_of::<AtomicUsize>()),
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

    #[cfg(debug_assertions)]
    fn check_ptr(&self, ptr: NonNull<u8>) -> bool {
        let addr = ptr.as_ptr().addr();
        let (origin, size) = self.chunk.pointer().to_raw_parts();
        let origin = origin.as_ptr().addr();
        (origin..origin + size).contains(&addr)
    }

    fn allocate(
        &self,
        thread_id: u64,
        count: usize,
        align: usize,
        is_large_or_huge: bool,
        base: &B,
    ) -> Option<Result<SlabRef, Error<B>>> {
        debug_assert!(align <= SLAB_SIZE);
        let ptr = self.allocate_slices(count)?;

        let (addr, _) = ptr.to_raw_parts();
        let commit = unsafe { base.commit(NonNull::from_raw_parts(addr, mem::size_of::<Slab>())) };
        // SAFETY: The fresh allocation is aligned to `SLAB_SIZE`.
        let res = commit.map(|_| unsafe {
            Slab::init(
                ptr,
                thread_id,
                self.arena_id,
                is_large_or_huge,
                B::IS_ZEROED,
            )
        });
        Some(res.map_err(Error::Commit))
    }

    /// # Safety
    ///
    /// - `slab` must be previously allocated from this arena;
    /// - No more references to the `slab` or its shards exist after calling
    ///   this function.
    unsafe fn deallocate(&self, slab: SlabRef, base: &B) -> usize {
        let raw = slab.into_raw();
        unsafe { base.decommit(raw) };
        let (ptr, len) = raw.to_raw_parts();
        let offset = unsafe { ptr.cast::<u8>().sub_ptr(self.chunk.pointer().cast()) };

        let (start, end) = (offset / SLAB_SIZE, (offset + len) / SLAB_SIZE);
        self.bitmap().set::<false>((start as u32)..(end as u32));
        end - start
    }
}

const MAX_ARENAS: usize = 112;
/// A collection of arenas.
///
/// This structure manages all the memory within its lifetime. Multiple
/// [`Context`](crate::heap::Context)s and [`Heap`](crate::heap::Heap)s can have
/// reference to one instance of this type.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Arenas<B: BaseAlloc> {
    pub(crate) base: B,
    arenas: [AtomicPtr<Arena<B>>; MAX_ARENAS],
    arena_count: AtomicUsize,
    // slab_count: AtomicUsize,
    abandoned: AtomicPtr<()>,
}

impl<B: BaseAlloc> Arenas<B> {
    // We're using this constant to initialize the array, so no real manipulation on
    // this constant is performed.
    #[allow(clippy::declare_interior_mutable_const)]
    const ARENA_INIT: AtomicPtr<Arena<B>> = AtomicPtr::new(ptr::null_mut());

    /// Creates a new collection of arenas.
    pub const fn new(base: B) -> Self {
        Arenas {
            base,
            arenas: [Self::ARENA_INIT; MAX_ARENAS],
            arena_count: AtomicUsize::new(0),
            // slab_count: AtomicUsize::new(0),
            abandoned: AtomicPtr::new(ptr::null_mut()),
        }
    }

    fn push_arena<'a>(&'a self, arena: &'a mut Arena<B>) -> Result<&'a Arena<B>, Error<B>> {
        let mut index = self.arena_count.load(Relaxed);
        loop {
            break if index < MAX_ARENAS {
                if let Err(i) = self
                    .arena_count
                    .compare_exchange(index, index + 1, AcqRel, Acquire)
                {
                    index = i;
                    continue;
                }
                if self.arenas[index]
                    .compare_exchange(ptr::null_mut(), arena, AcqRel, Acquire)
                    .is_err()
                {
                    index = self.arena_count.load(Relaxed);
                    continue;
                }
                arena.arena_id = index + 1;
                Ok(arena)
            } else if let Some(index) = self.arenas.iter().position(|slot| {
                slot.compare_exchange(ptr::null_mut(), arena, AcqRel, Acquire)
                    .is_ok()
            }) {
                arena.arena_id = index + 1;
                Ok(arena)
            } else {
                #[cfg(not(err_on_exhaustion))]
                panic!("ARENA EXHAUSTED");
                #[cfg(err_on_exhaustion)]
                {
                    // SAFETY: The arena is freshly allocated.
                    unsafe { Arena::drop(arena.into()) };
                    Err(Error::ArenaExhausted)
                }
            };
        }
    }

    fn push_abandoned(&self, slab: SlabRef) {
        debug_assert!(slab.is_abandoned());
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

    fn all_arenas(&self) -> impl Iterator<Item = (usize, &Arena<B>)> {
        let iter = self.arenas[..self.arena_count.load(Acquire)].iter();
        // SAFETY: We check the nullity of the pointers.
        iter.enumerate()
            .filter_map(|(index, arena)| Some((index, unsafe { arena.load(Acquire).as_ref() }?)))
    }

    fn arenas(&self, is_exclusive: bool) -> impl Iterator<Item = (usize, &Arena<B>)> {
        self.all_arenas()
            .filter(move |(_, arena)| arena.is_exclusive == is_exclusive)
    }

    /// Retrieves the base allocator of this arena collection.
    pub fn base(&self) -> &B {
        &self.base
    }

    /// Manages another chunk previously allocated by an instance of its base
    /// allocator.
    ///
    /// This function creates a new arena from the chunk and push it to the
    /// collection for further allocation, extending the heap's overall
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the header allocation has failed, or the collection
    /// is full of arenas.
    pub fn manage(&self, chunk: Chunk<B>) -> Result<(), Error<B>> {
        let arena = Arena::new_chunk(&self.base, chunk)?;
        self.push_arena(arena)?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    pub(crate) fn check_ptr(&self, ptr: NonNull<u8>) -> bool {
        self.all_arenas().any(|(_, arena)| arena.check_ptr(ptr))
    }

    pub(crate) fn allocate(
        &self,
        thread_id: u64,
        count: NonZeroUsize,
        align: usize,
        is_large_or_huge: bool,
    ) -> Result<SlabRef, Error<B>> {
        let count = count.get();

        let mut retry = 0;
        let ret = loop {
            self.collect_abandoned();
            match self.arenas(false).find_map(|(_, arena)| {
                arena.allocate(thread_id, count, align, is_large_or_huge, &self.base)
            }) {
                Some(slab) => break slab?,
                None if retry < 3 => retry += 1,
                None => {
                    const MIN_RESERVE_COUNT: usize = 32;

                    let reserve_count = count.max(MIN_RESERVE_COUNT)
                        /* .max(self.slab_count.load(Relaxed).isqrt()) */;
                    let arena = Arena::new(&self.base, reserve_count, Some(align), false)?;
                    let arena = self.push_arena(arena)?;
                    let res = arena.allocate(thread_id, count, align, is_large_or_huge, &self.base);
                    break res.unwrap()?;
                }
            }
        };
        // self.slab_count.fetch_add(count, Relaxed);
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
            let _slab_count = unsafe { (*arena).deallocate(slab, &self.base) };
            // self.slab_count.fetch_sub(slab_count, Relaxed);
        } else {
            self.push_abandoned(slab)
        }
    }

    /// Allocates a block of memory using `layout` from this structure directly,
    /// bypassing any instance of [`Heap`](crate::heap::Heap).
    ///
    /// The pointer must not be deallocated by any instance of `Heap` or other
    /// instances of this type.
    pub fn allocate_direct(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<B>> {
        let arena = Arena::new(
            &self.base,
            layout.size().div_ceil(SLAB_SIZE),
            Some(layout.align()),
            true,
        )?;
        let arena = self.push_arena(arena)?;
        Ok(NonNull::from_raw_parts(
            arena.chunk.pointer().cast(),
            layout.size(),
        ))
        .inspect(|&ptr| track::allocate(ptr, 0, false))
    }

    /// Retrieves the layout information of an allocation.
    ///
    /// If `ptr` was not previously allocated directly from this structure,
    /// `None` is returned.
    pub fn layout_of_direct(&self, ptr: NonNull<u8>) -> Option<Layout> {
        self.arenas(true)
            .find(|(_, arena)| arena.chunk.pointer().cast() == ptr)
            .map(|(_, arena)| arena.chunk.layout())
    }

    /// Deallocates an allocation previously from this structure.
    ///
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
            track::deallocate(ptr, 0);
            let arena = self.arenas[index].swap(ptr::null_mut(), AcqRel);
            debug_assert!(!arena.is_null());
            // SAFETY: `arena` is exclusive.
            unsafe { Arena::drop(NonNull::new_unchecked(arena)) }
        } else {
            #[cfg(debug_assertions)]
            panic!("deallocating memory not from these arenas: {ptr:p}")
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

/// The errors of all the functions of this crate.
#[derive(Debug)]
pub enum Error<B: BaseAlloc> {
    /// The base error returned when an allocation failed.
    Alloc(B::Error),
    /// The base error returned when an commission failed.
    Commit(B::Error),
    /// The arena collection is full of arenas.
    ArenaExhausted,
}

impl<B: BaseAlloc> core::fmt::Display for Error<B>
where
    B::Error: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Alloc(err) => write!(f, "base allocation failed: {err}"),
            Error::Commit(err) => write!(f, "base commission failed: {err}"),
            Error::ArenaExhausted => write!(f, "the arena collection is full of arenas"),
        }
    }
}
