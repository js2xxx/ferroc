//! The module of heaps and contexts.
//!
//! See [`Heap`] and [`Context`] for more information.

#[cfg(feature = "global")]
mod thread_local;

#[cfg(feature = "stat")]
use core::cell::RefCell;
use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    mem::MaybeUninit,
    num::NonZeroUsize,
    ptr::{self, NonNull},
    sync::atomic::{AtomicU64, Ordering::*},
};

#[cfg(feature = "global")]
pub use self::thread_local::{ThreadData, ThreadLocal};
use crate::{
    arena::{Arenas, Error, SHARD_SIZE, SLAB_SIZE},
    base::BaseAlloc,
    slab::{BlockRef, Shard, ShardFlags, ShardList, Slab, EMPTY_SHARD},
    track, Stat,
};

/// The count of small & medium object sizes.
pub const OBJ_SIZE_COUNT: usize = obj_size_index(ObjSizeType::LARGE_MAX) + 1;
#[cfg(feature = "finer-grained")]
pub const GRANULARITY_SHIFT: u32 = 3;
#[cfg(not(feature = "finer-grained"))]
pub const GRANULARITY_SHIFT: u32 = 4;
pub const GRANULARITY: usize = 1 << GRANULARITY_SHIFT;

const DIRECT_COUNT: usize = (ObjSizeType::SMALL_MAX >> GRANULARITY_SHIFT) + 1;

const fn direct_index(size: usize) -> usize {
    (size + GRANULARITY - 1) >> GRANULARITY_SHIFT
}

// const fn direct_size(direct_index: usize) -> usize {
//     direct_index << GRANULARITY_SHIFT
// }

/// Gets the index of a specific object size.
///
/// This function is the inverse function of [`obj_size`].
pub const fn obj_size_index(size: usize) -> usize {
    match size - 1 {
        #[cfg(feature = "finer-grained")]
        size_m1 @ 0..=63 => size_m1 >> 3,
        #[cfg(not(feature = "finer-grained"))]
        size_m1 @ 0..=63 => size_m1 >> 4,
        size_m1 => {
            let msb_m2 = (usize::BITS - size_m1.leading_zeros() - 3) as usize;

            #[cfg(feature = "finer-grained")]
            return ((msb_m2 - 2) << 2) + ((size_m1 >> msb_m2) & 3);
            #[cfg(not(feature = "finer-grained"))]
            {
                ((msb_m2 - 3) << 2) + ((size_m1 >> msb_m2) & 3)
            }
        }
    }
}

/// Gets the maximum object size of an object index.
///
/// This function is the inverse function of [`obj_size_index`].
pub const fn obj_size(index: usize) -> usize {
    #[cfg(feature = "finer-grained")]
    return match index {
        0..=6 => (index + 1) << 3,
        i => (64 + (((i - 7) & 3) << 4)) << ((i - 7) >> 2),
    };
    #[cfg(not(feature = "finer-grained"))]
    match index {
        0..=2 => (index + 1) << 4,
        i => (64 + (((i - 3) & 3) << 4)) << ((i - 3) >> 2),
    }
}

/// Gets the type of a specific object size.
pub const fn obj_size_type(size: usize) -> ObjSizeType {
    match size {
        s if s <= ObjSizeType::SMALL_MAX => Small,
        s if s <= ObjSizeType::MEDIUM_MAX => Medium,
        s if s <= ObjSizeType::LARGE_MAX => Large,
        _ => Huge,
    }
}

/// The type of object sizes.
pub enum ObjSizeType {
    Small,
    Medium,
    Large,
    Huge,
}
use array_macro::array;
use ObjSizeType::*;

impl ObjSizeType {
    pub const SMALL_MAX: usize = 1024;
    pub const MEDIUM_MAX: usize = SHARD_SIZE;
    pub const LARGE_MAX: usize = SLAB_SIZE / 2;
}

/// A memory allocator context of ferroc.
///
/// This structure serves as the heap storage of a specific task. It contains a
/// reference to [a collection of arenas](Arenas) so as to avoid the overhead of
/// `Arc<T>`.
///
/// Every context has its unique ID, which is for allocations to distinguish the
/// source.
///
/// Contexts cannot allocate memory directly. A [`Heap`] should be created from
/// this type and be used instead.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Context<'arena, B: BaseAlloc> {
    thread_id: u64,
    arena: &'arena Arenas<B>,
    free_shards: ShardList<'arena>,
    heap_count: Cell<usize>,
    #[cfg(feature = "stat")]
    stat: RefCell<Stat>,
}

impl<'arena, B: BaseAlloc> Context<'arena, B> {
    /// Creates a new context from a certain arena collection.
    ///
    /// Unlike [`Arenas::new`], this function cannot be called during constant
    /// evaluation.
    pub fn new(arena: &'arena Arenas<B>) -> Self {
        static ID: AtomicU64 = AtomicU64::new(1);
        // SAFETY: ID is unique.
        unsafe { Self::new_with_id(arena, ID.fetch_add(1, Relaxed)) }
    }

    /// # Safety
    ///
    /// `thread_id` must be unique with each live thread.
    unsafe fn new_with_id(arena: &'arena Arenas<B>, thread_id: u64) -> Self {
        Context {
            thread_id,
            arena,
            free_shards: Default::default(),
            heap_count: Cell::new(0),
            #[cfg(feature = "stat")]
            stat: RefCell::new(Stat::INIT),
        }
    }

    fn alloc_slab(
        &self,
        count: NonZeroUsize,
        align: usize,
        is_large_or_huge: bool,
        _stat: &mut Stat,
    ) -> Result<&'arena Shard<'arena>, Error<B>> {
        let slab = self
            .arena
            .allocate(self.thread_id, count, align, is_large_or_huge)?;
        #[cfg(feature = "stat")]
        {
            _stat.slabs += 1;
        }
        Ok(slab.into_shard())
    }

    fn finalize_shard(&self, shard: &'arena Shard<'arena>, stat: &mut Stat) {
        match shard.fini(stat) {
            Ok(Some(fini)) => {
                #[cfg(feature = "stat")]
                {
                    stat.free_shards += fini.shard_count();
                }
                self.free_shards.push(fini);
            }
            Ok(None) => {} // `slab` has abandoned shard(s), so we cannot reuse it.
            // `slab` is unused/abandoned, we can deallocate it.
            Err(slab) => {
                if !slab.is_large_or_huge {
                    let _count = self
                        .free_shards
                        .drain(|s| ptr::eq(s.slab().0, &*slab))
                        .fold(0, |a, s| a + s.shard_count());
                    #[cfg(feature = "stat")]
                    {
                        stat.free_shards -= _count;
                        if slab.is_abandoned() {
                            stat.abandoned_slabs += 1;
                        } else {
                            stat.slabs -= 1;
                        }
                    }
                }
                // SAFETY: All slabs are allocated from `self.arena`.
                unsafe { self.arena.deallocate(slab) }
            }
        }
    }
}

impl<'arena, B: BaseAlloc> Drop for Context<'arena, B> {
    fn drop(&mut self) {
        debug_assert!(self.free_shards.is_empty());
        debug_assert_eq!(self.heap_count.get(), 0);
        #[cfg(all(debug_assertions, feature = "stat"))]
        self.stat.get_mut().assert_clean();
        #[cfg(all(test, feature = "stat"))]
        std::println!("{}: {:?}", self.thread_id, self.stat.get_mut());
    }
}

struct Bin<'arena> {
    list: ShardList<'arena>,
    obj_size: usize,
    index: usize,
}

/// A memory allocator unit of ferroc.
///
/// This type serves as the most direct interface exposed to users compared with
/// other intermediate structures. Users usually allocate memory from this type.
///
/// By far, only 1 heap may exist from 1 context.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Heap<'arena: 'cx, 'cx, B: BaseAlloc> {
    cx: Option<&'cx Context<'arena, B>>,
    direct_shards: [Cell<&'arena Shard<'arena>>; DIRECT_COUNT],
    shards: [Bin<'arena>; OBJ_SIZE_COUNT],
    full_shards: ShardList<'arena>,
    huge_shards: ShardList<'arena>,
}

#[inline]
pub(crate) fn post_alloc(ptr: NonNull<[u8]>, is_zeroed: bool, zero: bool) {
    if zero && !is_zeroed {
        unsafe { ptr.as_uninit_slice_mut().fill(MaybeUninit::zeroed()) };
    }
    track::allocate(ptr, 0, zero);
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Heap<'arena, 'cx, B> {
    /// Creates a new heap from a memory allocator context.
    #[inline]
    pub fn new(cx: &'cx Context<'arena, B>) -> Self {
        let mut heap = Self::new_uninit();
        unsafe { heap.init(cx) };
        heap
    }

    /// Creates en empty (uninitialized) heap.
    ///
    /// This function should be paired with [`init`](Heap::init) to utilize its
    /// basic functionality, for the potential requirements for lazy
    /// initialization.
    pub const fn new_uninit() -> Self {
        Heap {
            cx: None,
            direct_shards: array![
                _ => Cell::new(EMPTY_SHARD.as_ref());
                DIRECT_COUNT
            ],
            shards: array![
                index => Bin {
                    list: ShardList::DEFAULT,
                    obj_size: obj_size(index),
                    index,
                };
                OBJ_SIZE_COUNT
            ],
            full_shards: ShardList::DEFAULT,
            huge_shards: ShardList::DEFAULT,
        }
    }

    /// Tests if this heap is initialized (bound to a [context](Context)).
    #[inline]
    pub const fn is_init(&self) -> bool {
        self.cx.is_some()
    }

    /// Tests if this heap is bound to a specific [context](Context).
    #[inline]
    pub fn is_bound_to(&self, cx: &Context<'arena, B>) -> bool {
        self.cx.is_some_and(|this| ptr::eq(this, cx))
    }

    /// Initializes this heap, binding it to a [context](Context).
    ///    
    /// This function should be paired with [`new_uninit`](Heap::new_uninit) to
    /// utilize its basic functionality, for the potential requirements for
    /// lazy initialization.
    ///
    /// # Safety
    ///
    /// This function must be called only once for every uninitialized heap, and
    /// must not be called for those created with [`new`](Heap::new)
    /// (initialized upon creation).
    #[inline]
    pub unsafe fn init(&mut self, cx: &'cx Context<'arena, B>) {
        const MAX_HEAP_COUNT: usize = 1;

        let count = cx.heap_count.get() + 1;
        assert!(
            count <= MAX_HEAP_COUNT,
            "a context can only have at most {MAX_HEAP_COUNT} heap(s)"
        );
        cx.heap_count.set(count);
        self.cx = Some(cx);
    }

    #[inline]
    fn cx(&self) -> Result<&'cx Context<'arena, B>, Error<B>> {
        self.cx.ok_or(Error::Uninit)
    }

    fn pop_huge_untracked(
        &self,
        size: usize,
        stat: &mut Stat,
        set_align: bool,
    ) -> Result<(NonNull<[u8]>, bool), Error<B>> {
        debug_assert!(size > ObjSizeType::LARGE_MAX);
        let cx = self.cx()?;

        let count = (Slab::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
        let count = NonZeroUsize::new(count).unwrap();
        let shard = cx.alloc_slab(count, SHARD_SIZE, true, stat)?;
        shard.init_large_or_huge(size, count, cx.arena.base(), stat)?;
        self.huge_shards.push(shard);

        let (block, is_zeroed) = shard.pop_block().unwrap();
        #[cfg(feature = "stat")]
        {
            stat.huge_count += 1;
            stat.huge_size += size;
        }
        if set_align {
            shard.flags.set_align();
        }
        Ok((NonNull::from_raw_parts(block.into_raw(), size), is_zeroed))
    }

    #[inline]
    fn pop_huge(
        &self,
        size: usize,
        zero: bool,
        stat: &mut Stat,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        let (ptr, is_zeroed) = self.pop_huge_untracked(size, stat, false)?;
        post_alloc(ptr, is_zeroed, zero);
        Ok(ptr)
    }

    fn update_direct(&self, bin: &Bin<'arena>) {
        if bin.obj_size > ObjSizeType::SMALL_MAX {
            return;
        }
        let shard = bin.list.current().unwrap_or(EMPTY_SHARD.as_ref());

        let direct_index = direct_index(bin.obj_size);
        if ptr::eq(self.direct_shards[direct_index].get(), shard) {
            return;
        }
        let end = direct_index;
        let start = if end == 1 {
            0
        } else {
            let obj_size = self::obj_size(bin.index - 1);
            let direct_index = self::direct_index(obj_size);
            direct_index + 1
        };
        let range = self.direct_shards[start..=end].iter();
        range.for_each(|d| d.set(shard));
    }

    #[inline]
    fn pop(&self, size: usize, zero: bool, stat: &mut Stat) -> Result<NonNull<[u8]>, Error<B>> {
        let (ptr, is_zeroed) = self.pop_untracked(size, stat, false)?;
        post_alloc(ptr, is_zeroed, zero);
        Ok(ptr)
    }

    fn pop_untracked(
        &self,
        size: usize,
        stat: &mut Stat,
        set_align: bool,
    ) -> Result<(NonNull<[u8]>, bool), Error<B>> {
        if size <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(size)
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some((block, is_zeroed)) = shard.pop_block()
        {
            #[cfg(feature = "stat")]
            {
                stat.direct_count += 1;
                stat.direct_size += size;
            }
            if set_align {
                shard.flags.set_align();
            }
            return Ok((NonNull::from_raw_parts(block.into_raw(), size), is_zeroed));
        }
        self.pop_contended(size, stat, set_align)
    }

    fn find_free_from_all(&self, bin: &Bin<'arena>, _stat: &mut Stat) -> Option<&Shard<'arena>> {
        let mut cursor = bin.list.cursor_head();
        loop {
            let shard = cursor.get()?;
            if !shard.collect(false) {
                shard.extend(bin.obj_size);
            }

            if shard.has_free() {
                return Some(shard);
            }

            cursor.remove();
            self.update_direct(bin);

            shard.flags.set_in_full(true);
            self.full_shards.push(shard);
        }
    }

    fn find_free(&self, bin: &Bin<'arena>, _stat: &mut Stat) -> Option<&Shard<'arena>> {
        if let Some(shard) = bin.list.current()
            && shard.collect(false)
        {
            return Some(shard);
        }
        self.find_free_from_all(bin, _stat)
    }

    #[cold]
    fn pop_contended(
        &self,
        size: usize,
        stat: &mut Stat,
        set_align: bool,
    ) -> Result<(NonNull<[u8]>, bool), Error<B>> {
        let is_large = match obj_size_type(size) {
            Huge => return self.pop_huge_untracked(size, stat, set_align),
            ty => matches!(ty, Large),
        };
        let cx = self.cx()?;
        let bin = &self.shards[obj_size_index(size.max(1))];

        if !bin.list.is_empty()
            && let Some(shard) = self.find_free(bin, stat)
        {
            debug_assert!(shard.has_free());
            // SAFETY: `shard` has free blocks.
            let (block, is_zeroed) = unsafe { shard.pop_block().unwrap_unchecked() };
            return Ok((NonNull::from_raw_parts(block.into_raw(), size), is_zeroed));
        }

        let fresh = if !is_large && let Some(free) = cx.free_shards.pop() {
            // 1. Try to pop from the free shards;
            free
        } else {
            // 2. Try to clear abandoned huge shards and allocate/reclaim a slab.
            // SAFETY: The current function needs the heap to be initialized.
            unsafe { self.collect_huge(stat) };
            cx.alloc_slab(NonZeroUsize::MIN, 1, is_large, stat)?
        };

        #[cfg(feature = "stat")]
        {
            stat.free_shards -= free.shard_count();
        }
        if let Some(next) = fresh.init(bin.obj_size, cx.arena.base(), stat)? {
            cx.free_shards.push(next);
            #[cfg(feature = "stat")]
            {
                stat.free_shards += next.shard_count();
            }
        }
        bin.list.push(fresh);
        self.update_direct(bin);

        debug_assert!(fresh.has_free());
        // SAFETY: `shard` has free blocks.
        let (block, is_zeroed) = unsafe { fresh.pop_block().unwrap_unchecked() };
        Ok((NonNull::from_raw_parts(block.into_raw(), size), is_zeroed))
    }

    fn pop_aligned(
        &self,
        layout: Layout,
        zero: bool,
        stat: &mut Stat,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.size() <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(layout.size())
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some((block, is_zeroed)) = shard.pop_block_aligned(layout.align())
        {
            #[cfg(feature = "stat")]
            {
                stat.direct_count += 1;
                stat.direct_size += layout.size();
            }
            let ptr = NonNull::from_raw_parts(block.into_raw(), layout.size());
            post_alloc(ptr, is_zeroed, zero);
            return Ok(ptr);
        }
        self.pop_aligned_contended(layout, zero, stat)
    }

    #[cold]
    fn pop_aligned_contended(
        &self,
        layout: Layout,
        zero: bool,
        stat: &mut Stat,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.size() <= ObjSizeType::LARGE_MAX
            && let index = obj_size_index(layout.size())
            && let Some(shard) = self.shards[index].list.current()
            && let Some((block, is_zeroed)) = shard.pop_block_aligned(layout.align())
        {
            #[cfg(feature = "stat")]
            {
                stat.normal_count[index] += 1;
                stat.normal_size += obj_size(index);
            }
            let ptr = NonNull::from_raw_parts(block.into_raw(), layout.size());
            post_alloc(ptr, is_zeroed, zero);
            Ok(ptr)
        } else if layout.align() <= SHARD_SIZE && layout.size() > ObjSizeType::LARGE_MAX {
            self.pop_huge(layout.size(), zero, stat)
        } else if layout.align() <= ObjSizeType::LARGE_MAX {
            let (ptr, is_zeroed) =
                self.pop_untracked(layout.size() + layout.align() - 1, stat, true)?;
            let addr = (ptr.addr().get() + layout.align() - 1) & !(layout.align() - 1);
            let ptr = ptr.cast().with_addr(NonZeroUsize::new(addr).unwrap());

            let ptr = NonNull::from_raw_parts(ptr, layout.size());
            post_alloc(ptr, is_zeroed, zero);
            Ok(ptr)
        } else {
            Err(Error::Unsupported(layout))
        }
    }

    unsafe fn allocate_inner(&self, layout: Layout, zero: bool) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.size() == 0 {
            return Ok(NonNull::from_raw_parts(layout.dangling().cast(), 0));
        }
        #[cfg(feature = "stat")]
        let mut stat = self.cx()?.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();
        if layout.size() <= ObjSizeType::MEDIUM_MAX && layout.size() & (layout.align() - 1) == 0 {
            return (self.pop(layout.size(), zero, &mut stat))
                .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())));
        }
        self.pop_aligned(layout, zero, &mut stat)
            .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())))
    }

    #[cfg(feature = "c")]
    pub(crate) fn malloc(&self, size: usize, zero: bool) -> Result<NonNull<[u8]>, Error<B>> {
        #[cfg(feature = "stat")]
        let mut stat = self.cx()?.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();
        self.pop(size, zero, &mut stat)
    }

    /// Allocate a memory block of `layout`.
    ///
    /// The allocation can be deallocated by other heaps referring to the same
    /// arena collection.
    ///
    /// # Errors
    ///
    /// Errors are returned when allocation fails, see [`Error`] for more
    /// information.
    #[inline]
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<B>> {
        // SAFETY: `cx` is initialized.
        unsafe { self.allocate_inner(layout, false) }
    }

    /// Allocate a zeroed memory block of `layout`.
    ///
    /// The allocation can be deallocated by other heaps referring to the same
    /// arena collection.
    ///
    /// # Errors
    ///
    /// Errors are returned when allocation fails, see [`Error`] for more
    /// information.
    #[inline]
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<B>> {
        // SAFETY: `cx` is initialized.
        unsafe { self.allocate_inner(layout, true) }
    }

    /// Retrives the statstics of this heap.
    ///
    /// The statistics of all heaps cannot be retrieved directly. Users should
    /// calculate it from this function.
    #[cfg(feature = "stat")]
    pub fn stat(&self) -> Stat {
        *self.cx.unwrap().stat.borrow()
    }

    /// Retrieves the layout information of a specific allocation.
    ///
    /// The layout returned may not be the same of the layout passed to
    /// [`allocate`](Heap::allocate), but is the most fit layout of it, and can
    /// be passed to [`deallocate`](Heap::deallocate).
    ///
    /// # Errors
    ///
    /// [`Error::Uninit`] is returned if the heap needs to be initialized first
    /// before retrieving the layout of this allocation.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to an owned, valid memory block of `layout`,
    ///   previously allocated by a certain instance of `Heap` alive in the
    ///   scope, created from the same arena.
    /// - The allocation size must not be 0.
    pub unsafe fn layout_of(&self, ptr: NonNull<u8>) -> Layout {
        #[cfg(debug_assertions)]
        if let Some(cx) = self.cx
            && !cx.arena.check_ptr(ptr)
        {
            unreachable!("{ptr:p} is not allocated from these arenas");
        }
        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let slab = unsafe { Slab::from_ptr(ptr).unwrap_unchecked() };
        // SAFETY: The same as `shard_meta`.
        let shard = Slab::shard_meta(slab, ptr.cast());
        let obj_size = unsafe { Shard::obj_size_raw(shard) };
        // SAFETY: `ptr` is in `shard`.
        let block = unsafe { Shard::block_of(shard, ptr.cast()) };

        let size = obj_size - (ptr.addr().get() - block.into_raw().addr().get());
        let align = 1 << ptr.addr().get().trailing_zeros();
        Layout::from_size_align(size, align).unwrap()
    }

    /// Deallocates an allocation previously allocated by an instance of heap
    /// referring to the same collection of arenas.
    ///
    /// # Errors
    ///
    /// [`Error::Uninit`] is returned if the heap needs to be initialized first
    /// before deallocating this pointer.
    ///
    /// The corresponding implementation in `core::alloc::Allocator` will
    /// silently leak the allocation instead in this case.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to an owned, valid memory block of `layout`,
    ///   previously allocated by a certain instance of `Heap` alive in the
    ///   scope, created from the same arena.
    /// - No aliases of `ptr` should exist after the deallocation.
    #[inline]
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        #[cfg(debug_assertions)]
        {
            let tested_layout = self.layout_of(ptr);
            debug_assert!(tested_layout.size() >= layout.size());
            debug_assert!(tested_layout.align() >= layout.align());
        }
        // SAFETY: `ptr` is allocated by these structures.
        unsafe { self.free(ptr) }
    }

    unsafe fn free_shard(&self, shard: &'arena Shard<'arena>, obj_size: usize, stat: &mut Stat) {
        debug_assert!(shard.is_unused());
        let cx = self.cx.unwrap_unchecked();

        debug_assert_eq!(obj_size, shard.obj_size.load(Relaxed));
        if obj_size <= ObjSizeType::LARGE_MAX {
            let index = obj_size_index(obj_size);
            let bin = &self.shards[index];
            #[cfg(feature = "stat")]
            {
                stat.normal_count[index] -= 1;
                stat.normal_size -= self::obj_size(index);
            }

            if bin.list.len() > 1 {
                let _ret = bin.list.remove(shard);
                self.update_direct(bin);
                debug_assert!(_ret);
                cx.finalize_shard(shard, stat);
            }
        } else {
            #[cfg(feature = "stat")]
            {
                stat.huge_count -= 1;
                stat.huge_size -= obj_size;
            }

            let _ret = self.huge_shards.remove(shard);
            debug_assert!(_ret);
            cx.finalize_shard(shard, stat);
        }
    }

    /// # Errors
    ///
    /// [`Error::Uninit`] is returned if the heap needs to be initialized first
    /// before deallocating this pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to an owned, valid memory block, previously allocated
    ///   by a certain instance of `Heap` alive in the scope, created from the
    ///   same arena.
    /// - No aliases of `ptr` should exist after the deallocation.
    /// - The allocation size must not be 0.
    pub(crate) unsafe fn free(&self, ptr: NonNull<u8>) {
        #[cfg(debug_assertions)]
        if let Some(cx) = self.cx
            && !cx.arena.check_ptr(ptr)
        {
            unreachable!("{ptr:p} is not allocated from these arenas");
        }
        track::deallocate(ptr, 0);

        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let slab = unsafe { Slab::from_ptr(ptr).unwrap_unchecked() };
        let shard = unsafe { Slab::shard_meta(slab, ptr.cast()) };

        let thread_id = unsafe { ptr::addr_of!((*slab.as_ptr()).thread_id).read() };
        debug_assert_ne!(thread_id, 0);

        if let Some(cx) = self.cx
            && cx.thread_id == thread_id
        {
            if ShardFlags::test_zero(ptr::addr_of!((*shard.as_ptr()).flags)) {
                #[cfg(feature = "stat")]
                let mut stat = cx.stat.borrow_mut();
                #[cfg(not(feature = "stat"))]
                let mut stat = ();

                let shard = unsafe { shard.as_ref() };
                let block = unsafe { BlockRef::from_raw(ptr.cast()) };
                if shard.push_block(block) {
                    self.free_shard(shard, shard.obj_size.load(Relaxed), &mut stat)
                }
            } else {
                self.free_contended(ptr, shard, true)
            }
        } else {
            self.free_contended(ptr, shard, false)
        }
    }

    #[cold]
    pub(crate) unsafe fn free_contended(
        &self,
        ptr: NonNull<u8>,
        shard: NonNull<Shard<'arena>>,
        is_local: bool,
    ) {
        // SAFETY: `ptr` is in `shard`.
        let block = unsafe { Shard::block_of(shard, ptr.cast()) };

        if is_local {
            // `thread_id` matches; We're deallocating from the same thread.
            #[cfg(feature = "stat")]
            let mut stat = cx.stat.borrow_mut();
            #[cfg(not(feature = "stat"))]
            let mut stat = ();

            let shard = unsafe { shard.as_ref() };

            let was_full = shard.flags.is_in_full();
            let is_unused = shard.push_block(block);

            if is_unused || was_full {
                let obj_size = shard.obj_size.load(Relaxed);

                if was_full {
                    let index = obj_size_index(obj_size);
                    let bin = &self.shards[index];

                    let _ret = self.full_shards.remove(shard);
                    debug_assert!(_ret);
                    shard.flags.set_in_full(false);
                    bin.list.push(shard);
                    self.update_direct(bin);
                }

                if is_unused {
                    self.free_shard(shard, obj_size, &mut stat)
                }
            }
        } else {
            // We're deallocating from another thread.
            unsafe { Shard::push_block_mt(shard, block) }
        }
    }

    /// # Safety
    ///
    /// The heap must be initialized.
    unsafe fn collect_huge(&self, stat: &mut Stat) {
        let huge = self.huge_shards.drain(|shard| {
            shard.collect(false);
            shard.is_unused()
        });
        huge.for_each(|shard| {
            let cx = unsafe { self.cx.unwrap_unchecked() };
            cx.finalize_shard(shard, stat)
        });
    }

    /// Clean up some garbage data immediately.
    ///
    /// In short, due to implementation details, the free list (i.e. the popper
    /// of allocation) and the deallocation list (i.e. the pusher of
    /// deallocation) are 2 distinct lists.
    ///
    /// Those 2 lists will be swapped if the former becomes empty during the
    /// allocation process, which is precisely what this function does.
    pub fn collect(&self, force: bool) {
        let shards = self.shards.iter().flat_map(|bin| &bin.list);
        shards.for_each(|shard| {
            shard.collect(force);
        });

        #[cfg(feature = "stat")]
        let mut stat = self.cx.unwrap().stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();

        if self.is_init() {
            // SAFETY: `self` is initialized.
            unsafe { self.collect_huge(&mut stat) };
        }
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Drop for Heap<'arena, 'cx, B> {
    fn drop(&mut self) {
        if let Some(cx) = self.cx.take() {
            #[cfg(feature = "stat")]
            let mut stat = cx.stat.borrow_mut();
            #[cfg(not(feature = "stat"))]
            let mut stat = ();

            let iter = (self.shards.iter().map(|bin| &bin.list))
                .chain([&self.huge_shards, &self.full_shards]);
            iter.flat_map(|l| l.drain(|_| true)).for_each(|shard| {
                shard.collect(false);
                shard.flags.reset();
                cx.finalize_shard(shard, &mut stat);
            });
            cx.heap_count.set(cx.heap_count.get() - 1);
        }
    }
}

unsafe impl<'arena: 'cx, 'cx, B: BaseAlloc> Allocator for Heap<'arena, 'cx, B> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout).map_err(|_| AllocError)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate_zeroed(layout).map_err(|_| AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use crate::heap::{obj_size, obj_size_index, GRANULARITY};

    #[test]
    fn test_obj_size() {
        #[cfg(not(feature = "finer-grained"))]
        const SIZES: &[usize] = &[
            16, 32, 48, // A
            64, 80, 96, 112, // B1
            128, 160, 192, 224, // B2
            256, 320, 384, 448, // B3
        ];
        #[cfg(feature = "finer-grained")]
        const SIZES: &[usize] = &[
            8, 16, 24, 32, 40, 48, 56, // A
            64, 72, 80, 88, 96, 104, 112, 120, // B1
            128, 144, 160, 176, 192, 208, 224, 240, // B2
            256, 288, 320, 352, 384, 416, 448, 480, // B3
        ];
        for (index, &size) in SIZES.iter().enumerate() {
            assert_eq!(obj_size_index(size), index);
            assert_eq!(obj_size_index(size - GRANULARITY + 1), index);
            assert_eq!(obj_size(index), size);
        }
    }
}
