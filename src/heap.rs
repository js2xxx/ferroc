//! The module of heaps and contexts.
//!
//! See [`Heap`] and [`Context`] for more information.

#[cfg(feature = "global")]
mod thread_local;

use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    mem::MaybeUninit,
    num::NonZeroUsize,
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::Ordering::*,
};

use array_macro::array;

#[cfg(feature = "global")]
pub use self::thread_local::{ThreadData, ThreadLocal};
use crate::{
    arena::{Arenas, Error, SHARD_SIZE, SLAB_SIZE},
    base::BaseAlloc,
    slab::{BlockRef, Shard, ShardList, Slab, EMPTY_SHARD},
    track,
};

/// The count of small & medium object sizes.
pub const OBJ_SIZE_COUNT: usize = obj_size_index(ObjSizeType::LARGE_MAX) + 1;
/// The minimal alignment provided by this allocator, in bits.
#[cfg(feature = "finer-grained")]
pub const GRANULARITY_SHIFT: u32 = 3;
/// The minimal alignment provided by this allocator, in bits.
#[cfg(not(feature = "finer-grained"))]
pub const GRANULARITY_SHIFT: u32 = 4;
/// The minimal alignment provided by this allocator.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjSizeType {
    Small,
    Medium,
    Large,
    Huge,
}
use ObjSizeType::*;

impl ObjSizeType {
    /// The maximal size of small-sized objects.
    pub const SMALL_MAX: usize = 1024;
    /// The maximal size of medium-sized objects.
    pub const MEDIUM_MAX: usize = SHARD_SIZE / 2;
    /// The maximal size of larget-sized objects.
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
    arena: &'arena Arenas<B>,
    free_shards: ShardList<'arena>,
    heap_count: Cell<usize>,
}

impl<'arena, B: BaseAlloc> Context<'arena, B> {
    /// Creates a new context from a certain arena collection.
    ///
    /// Unlike [`Arenas::new`], this function cannot be called during constant
    /// evaluation.
    pub fn new(arena: &'arena Arenas<B>) -> Self {
        Context {
            arena,
            free_shards: Default::default(),
            heap_count: Cell::new(0),
        }
    }

    fn thread_id(self: Pin<&Self>) -> usize {
        (&*self as *const Self).addr()
    }

    fn alloc_slab(
        self: Pin<&Self>,
        count: NonZeroUsize,
        align: usize,
        is_large_or_huge: bool,
        direct: bool,
    ) -> Result<&'arena Shard<'arena>, Error<B>> {
        let slab = self
            .arena
            .allocate(self.thread_id(), count, align, is_large_or_huge, direct)?;
        Ok(slab.into_shard())
    }

    fn finalize_shard(self: Pin<&Self>, shard: &'arena Shard<'arena>) {
        match shard.fini() {
            Ok(Some(fini)) => {
                self.free_shards.push(fini);
            }
            Ok(None) => {} // `slab` has abandoned shard(s), so we cannot reuse it.
            // `slab` is unused/abandoned, we can deallocate it.
            Err(slab) => {
                if !slab.is_large_or_huge {
                    let _count = self
                        .free_shards
                        .drain(|s| ptr::eq(unsafe { s.slab::<B>().0 }, &*slab))
                        .fold(0, |a, s| a + s.shard_count());
                }
                // SAFETY: All slabs are allocated from `self.arena`.
                unsafe { self.arena.deallocate(slab) }
            }
        }
    }
}

impl<'arena, B: BaseAlloc> Drop for Context<'arena, B> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        debug_assert!(self.free_shards.is_empty());
        debug_assert_eq!(self.heap_count.get(), 0);
    }
}

struct Bin<'arena> {
    list: ShardList<'arena>,
    obj_size: usize,
    min_direct_index: usize,
    max_direct_index: usize,
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
    cx: Option<Pin<&'cx Context<'arena, B>>>,
    direct_shards: [Cell<&'arena Shard<'arena>>; DIRECT_COUNT],
    shards: [Bin<'arena>; OBJ_SIZE_COUNT],
    full_shards: ShardList<'arena>,
    huge_shards: ShardList<'arena>,
}

#[inline]
pub(crate) fn post_alloc(block: BlockRef, size: usize, is_zeroed: bool, zero: bool) -> NonNull<()> {
    let ptr = block.into_raw();
    let ptr_slice = NonNull::from_raw_parts(ptr, size);
    if zero && !is_zeroed {
        unsafe { ptr_slice.as_uninit_slice_mut().fill(MaybeUninit::zeroed()) };
    }
    track::allocate(ptr_slice, 0, zero);
    ptr
}

macro_rules! stry {
    ($e:expr, $s:expr) => {
        match $e {
            Ok(it) => it,
            Err(err) => {
                ($s)(err);
                return None;
            }
        }
    };
}

fn drop_<T>(_t: T) {}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Heap<'arena, 'cx, B> {
    /// Creates a new heap from a memory allocator context.
    #[inline]
    pub fn new(cx: Pin<&'cx Context<'arena, B>>) -> Self {
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
                    min_direct_index: match index.checked_sub(1) {
                        None => 0,
                        Some(index_m1) => direct_index(obj_size(index_m1)) + 1
                    },
                    max_direct_index: direct_index(obj_size(index)),
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
    pub fn is_bound_to(&self, cx: Pin<&Context<'arena, B>>) -> bool {
        self.cx.is_some_and(|this| ptr::eq(&*this, &*cx))
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
    pub unsafe fn init(&mut self, cx: Pin<&'cx Context<'arena, B>>) {
        const MAX_HEAP_COUNT: usize = 1;

        let count = cx.heap_count.get() + 1;
        assert!(
            count <= MAX_HEAP_COUNT,
            "a context can only have at most {MAX_HEAP_COUNT} heap(s)"
        );
        cx.heap_count.set(count);
        self.cx = Some(cx);
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Heap<'arena, 'cx, B> {
    fn pop_huge<'a>(
        &'a self,
        size: usize,
        set_align: bool,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        unsafe fn pop_huge_inner<B: BaseAlloc>(
            heap: &Heap<B>,
            size: usize,
            set_align: bool,
            zero: bool,
            sink: impl FnOnce(Error<B>),
        ) -> Option<NonNull<()>> {
            // SAFETY: The heap is initialized.
            let cx = unsafe { heap.cx.unwrap_unchecked() };

            let count = (Slab::<B>::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
            let count = NonZeroUsize::new(count).unwrap();
            let shard = stry!(cx.alloc_slab(count, SHARD_SIZE, true, true), sink);
            stry!(
                shard.init_large_or_huge(size, count, cx.arena.base(),),
                sink
            );
            heap.huge_shards.push(shard);

            let (block, is_zeroed) = shard.pop_block().unwrap();
            if set_align {
                shard.flags.set_align();
            }
            Some(post_alloc(block, size, is_zeroed, zero))
        }
        debug_assert!(size > ObjSizeType::LARGE_MAX);

        let heap = if !self.is_init() { fallback() } else { self };
        // SAFETY: The heap is initialized.
        unsafe { pop_huge_inner(heap, size, set_align, zero, err_sink) }
    }

    fn update_direct(&self, bin: &Bin<'arena>) {
        if bin.obj_size > ObjSizeType::SMALL_MAX {
            return;
        }
        let shard = bin.list.current().unwrap_or(EMPTY_SHARD.as_ref());
        if ptr::eq(self.direct_shards[bin.max_direct_index].get(), shard) {
            return;
        }
        let slice = &self.direct_shards[bin.min_direct_index..=bin.max_direct_index];
        slice.iter().for_each(|d| d.set(shard));
    }

    #[inline]
    fn pop<'a>(
        &'a self,
        size: usize,
        set_align: bool,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        if size <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(size)
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some((block, is_zeroed)) = shard.pop_block()
        {
            if set_align {
                shard.flags.set_align();
            }
            return Some(post_alloc(block, size, is_zeroed, zero));
        }
        let heap = if self.is_init() { self } else { fallback() };
        unsafe { heap.pop_contended(size, set_align, zero, err_sink) }
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    unsafe fn find_free_from_all(
        &self,
        bin: &Bin<'arena>,
        is_large: bool,
        first_try: bool,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<&Shard<'arena>> {
        let mut cursor = bin.list.cursor_head();
        while let Some(shard) = cursor.get() {
            if shard.collect(false) {
                return Some(shard);
            }
            if shard.extend(bin.obj_size) {
                return Some(shard);
            }

            cursor.move_next();
            
            shard.flags.set_in_full(true);
            bin.list.requeue_to(shard, &self.full_shards);
            self.update_direct(bin);
        }

        // SAFETY: `cx` is initialized.
        let cx = unsafe { self.cx.unwrap_unchecked() };
        macro_rules! add_fresh {
            ($fresh:ident) => {
                if let Some(next) = stry!($fresh.init(bin.obj_size, cx.arena.base()), err_sink) {
                    cx.free_shards.push(next);
                }
                bin.list.push($fresh);
                self.update_direct(bin);

                debug_assert!($fresh.has_free());
            };
        }

        // 1. Try to pop from the free shards;
        if !is_large && let Some(free) = cx.free_shards.pop() {
            add_fresh!(free);
            return Some(free);
        }

        // 2. Try to collect & unfull some shards (freed from other threads);
        let unfulled = self.full_shards.drain(|shard| shard.collect(false));
        if let Some(unfulled) = {
            unfulled.fold(None, |acc, shard| {
                shard.flags.set_in_full(false);

                let obj_size = shard.obj_size.load(Relaxed);
                let i = obj_size_index(obj_size);
                debug_assert!(i < OBJ_SIZE_COUNT);
                // SAFETY: i < OBJ_SIZE_COUNT.
                let unfulled_bin = unsafe { self.shards.get_unchecked(i) };

                unfulled_bin.list.push(shard);
                self.update_direct(unfulled_bin);
                acc.or_else(|| ptr::eq(bin, unfulled_bin).then_some(shard))
            })
        } {
            return Some(unfulled);
        }

        // 3. Try to clear abandoned huge shards and allocate/reclaim a slab.

        // SAFETY: The current function needs the heap to be initialized.
        unsafe { self.collect_huge() };
        let new = stry!(
            cx.alloc_slab(NonZeroUsize::MIN, 1, is_large, !first_try),
            err_sink
        );
        add_fresh!(new);
        Some(new)
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    unsafe fn find_free(
        &self,
        bin: &Bin<'arena>,
        is_large: bool,
        first_try: bool,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<&Shard<'arena>> {
        if let Some(shard) = bin.list.current()
            && shard.collect(false)
        {
            return Some(shard);
        }
        self.find_free_from_all(bin, is_large, first_try, err_sink)
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    #[cold]
    unsafe fn pop_contended(
        &self,
        size: usize,
        set_align: bool,
        zero: bool,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        let is_large = match obj_size_type(size) {
            Huge => return self.pop_huge(size, set_align, zero, || unreachable!(), err_sink),
            ty => matches!(ty, Large),
        };
        let index = obj_size_index(size.max(1));
        debug_assert!(index < OBJ_SIZE_COUNT);
        let bin = unsafe { self.shards.get_unchecked(index) };

        let shard = if let Some(shard) = self.find_free(bin, is_large, true, err_sink) {
            shard
        } else {
            self.collect(true);
            self.find_free(bin, is_large, false, |_| {})?
        };

        debug_assert!(shard.has_free());
        // SAFETY: `shard` has free blocks.
        let (block, is_zeroed) = unsafe { shard.pop_block().unwrap_unchecked() };
        Some(post_alloc(block, size, is_zeroed, zero))
    }

    fn pop_aligned<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        if layout.size() <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(layout.size())
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some((block, is_zeroed)) = shard.pop_block_aligned(layout.align())
        {
            return Some(post_alloc(block, layout.size(), is_zeroed, zero));
        }
        self.pop_aligned_contended(layout, zero, fallback, err_sink)
    }

    #[cold]
    fn pop_aligned_contended<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        if layout.size() <= ObjSizeType::LARGE_MAX
            && let index = obj_size_index(layout.size())
            // SAFETY: layout.size() <= ObjSizeType::LARGE_MAX means index < OBJ_SIZE_COUNT
            && let Some(shard) = unsafe { self.shards.get_unchecked(index) }.list.current()
            && let Some((block, is_zeroed)) = shard.pop_block_aligned(layout.align())
        {
            Some(post_alloc(block, layout.size(), is_zeroed, zero))
        } else if layout.align() <= SHARD_SIZE && layout.size() > ObjSizeType::LARGE_MAX {
            self.pop_huge(layout.size(), false, zero, fallback, err_sink)
        } else if layout.align() <= ObjSizeType::LARGE_MAX {
            let oversize = layout.size() + layout.align() - 1;
            let overptr = self.pop(oversize, true, zero, fallback, err_sink)?.cast();
            let addr = (overptr.addr().get() + layout.align() - 1) & !(layout.align() - 1);
            let ptr = overptr.with_addr(NonZeroUsize::new(addr).unwrap());
            if ptr.addr() != overptr.addr() {
                track::no_access(overptr.cast(), ptr.addr().get() - overptr.addr().get());
            }
            Some(ptr)
        } else {
            err_sink(Error::Unsupported(layout));
            None
        }
    }

    fn allocate_inner<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
        err_sink: impl FnOnce(Error<B>),
    ) -> Option<NonNull<()>> {
        if layout.size() == 0 {
            return Some(layout.dangling().cast());
        }
        if layout.size() <= ObjSizeType::MEDIUM_MAX && layout.size() & (layout.align() - 1) == 0 {
            return (self.pop(layout.size(), false, zero, fallback, err_sink))
                .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())));
        }
        self.pop_aligned(layout, zero, fallback, err_sink)
            .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())))
    }

    /// Get the default allocate options of the current heap.
    ///
    /// The default heap fallback will panic, while the default error sink will
    /// silently drop the error.
    #[allow(clippy::type_complexity)]
    pub fn options<'a>() -> AllocateOptions<fn() -> &'a Self, fn(Error<B>)> {
        AllocateOptions {
            fallback: || unreachable!("uninitialized heap"),
            error_sink: drop_,
        }
    }

    /// Allocate a memory block of `layout` with additional options.
    ///
    /// See [`AllocateOptions`] for the meaning of them.
    pub fn allocate_with<'a, F, E>(
        &'a self,
        layout: Layout,
        zero: bool,
        options: AllocateOptions<F, E>,
    ) -> Option<NonNull<()>>
    where
        F: FnOnce() -> &'a Self,
        E: FnOnce(Error<B>),
    {
        let fallback = options.fallback;
        let err_sink = options.error_sink;
        self.allocate_inner(layout, zero, fallback, err_sink)
    }

    #[cfg(feature = "c")]
    pub(crate) fn malloc<'a>(
        &'a self,
        size: usize,
        zero: bool,
        fallback: impl FnOnce() -> &'a Self,
    ) -> Option<NonNull<()>> {
        self.pop(size, false, zero, fallback, |_| {})
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
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<()>, Error<B>> {
        let mut err = MaybeUninit::uninit();
        let options = Self::options().error_sink(|e| drop_(err.write(e)));
        self.allocate_with(layout, false, options)
            .ok_or_else(|| unsafe { err.assume_init() })
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
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<()>, Error<B>> {
        let mut err = MaybeUninit::uninit();
        let options = Self::options().error_sink(|e| drop_(err.write(e)));
        self.allocate_with(layout, true, options)
            .ok_or_else(|| unsafe { err.assume_init() })
    }

    /// Retrieves the layout information of a specific allocation.
    ///
    /// The layout returned may not be the same of the layout passed to
    /// [`allocate`](Heap::allocate), but is the most fit layout of it, and can
    /// be passed to [`deallocate`](Heap::deallocate).
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
        let slab = unsafe { Slab::<B>::from_ptr(ptr).unwrap_unchecked() };
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

    unsafe fn free_shard(&self, shard: &'arena Shard<'arena>, obj_size: usize) {
        debug_assert!(shard.is_unused());
        let cx = self.cx.unwrap_unchecked();

        debug_assert_eq!(obj_size, shard.obj_size.load(Relaxed));
        if obj_size <= ObjSizeType::LARGE_MAX {
            let index = obj_size_index(obj_size);
            // SAFETY: layout.size() <= ObjSizeType::LARGE_MAX means index < OBJ_SIZE_COUNT.
            let bin = unsafe { self.shards.get_unchecked(index) };

            if !bin.list.has_sole_member() {
                bin.list.remove(shard);
                self.update_direct(bin);
                cx.finalize_shard(shard);
            }
        } else {
            self.huge_shards.remove(shard);
            cx.finalize_shard(shard);
        }
    }

    #[inline]
    fn thread_id(&self) -> usize {
        match self.cx {
            Some(ref cx) => cx.thread_id(),
            None => 0,
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

        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let slab = unsafe { Slab::<B>::from_ptr(ptr).unwrap_unchecked() };
        let shard = unsafe { Slab::shard_meta(slab, ptr.cast()) };

        let thread_id = unsafe { ptr::addr_of!((*slab.as_ptr()).thread_id).read() };
        debug_assert_ne!(thread_id, 0);

        if self.thread_id() == thread_id {
            // SAFETY: We're in the same thread.
            let shard = unsafe { shard.as_ref() };
            if shard.flags.test_zero() {
                track::deallocate(ptr, 0);

                // SAFETY: flags is zero, this shard has no aligned blocks.
                if shard.push_block(unsafe { BlockRef::from_raw(ptr.cast()) }) {
                    self.free_shard(shard, shard.obj_size.load(Relaxed))
                }
            } else {
                self.free_contended(ptr, shard.into(), true)
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
        track::deallocate(NonNull::new_unchecked(block.as_ptr().cast()), 0);
        if is_local {
            let shard = unsafe { shard.as_ref() };

            let is_unused = shard.push_block(block);
            if is_unused {
                self.free_shard(shard, shard.obj_size.load(Relaxed))
            } else if shard.flags.is_in_full() {
                debug_assert!(!is_unused);

                let obj_size = shard.obj_size.load(Relaxed);
                let index = obj_size_index(obj_size);

                shard.flags.set_in_full(false);

                debug_assert_ne!(obj_size_type(obj_size), ObjSizeType::Huge);
                // SAFETY: Huge shard cannot be full.
                let bin = unsafe { self.shards.get_unchecked(index) };
                self.full_shards.requeue_to(shard, &bin.list);
                self.update_direct(bin);
            }
        } else {
            // We're deallocating from another thread.
            unsafe { Shard::push_block_mt(shard, block) }
        }
    }

    /// # Safety
    ///
    /// The heap must be initialized.
    unsafe fn collect_huge(&self) {
        let huge = self.huge_shards.drain(|shard| {
            shard.collect(false);
            shard.is_unused()
        });
        huge.for_each(|shard| {
            let cx = unsafe { self.cx.unwrap_unchecked() };
            cx.finalize_shard(shard)
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

        if self.is_init() {
            // SAFETY: `self` is initialized.
            unsafe { self.collect_huge() };
        }

        if force && let Some(cx) = self.cx {
            cx.arena.reclaim_all();
        }
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Drop for Heap<'arena, 'cx, B> {
    fn drop(&mut self) {
        if let Some(cx) = self.cx.take() {
            let iter = (self.shards.iter().map(|bin| &bin.list))
                .chain([&self.huge_shards, &self.full_shards]);
            iter.flat_map(|l| l.drain(|_| true)).for_each(|shard| {
                shard.collect(false);
                shard.flags.reset();
                cx.finalize_shard(shard);
            });
            cx.heap_count.set(cx.heap_count.get() - 1);
        }
    }
}

unsafe impl<'arena: 'cx, 'cx, B: BaseAlloc> Allocator for Heap<'arena, 'cx, B> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self.allocate_with(layout, false, Self::options()) {
            Some(t) => Ok(NonNull::from_raw_parts(t, layout.size())),
            None => Err(AllocError),
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self.allocate_with(layout, true, Self::options()) {
            Some(t) => Ok(NonNull::from_raw_parts(t, layout.size())),
            None => Err(AllocError),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout);
    }
}

/// The additional allocate options which can be passed into
/// [`allocate_with`](Heap::allocate_with).
#[derive(Clone, Copy)]
pub struct AllocateOptions<F, E> {
    fallback: F,
    error_sink: E,
}

impl<F, E> AllocateOptions<F, E> {
    /// Creates a new allocate options.
    pub const fn new(fallback: F, error_sink: E) -> Self {
        AllocateOptions { fallback, error_sink }
    }

    /// Replace with a new fallback.
    ///
    /// The fallback returns lazyily evaluated backup heap when the current heap
    /// is not initialized (fails to acquire memory from its context).
    pub fn fallback<F2>(self, fallback: F2) -> AllocateOptions<F2, E> {
        AllocateOptions {
            fallback,
            error_sink: self.error_sink,
        }
    }

    /// Replace with a new error sink.
    ///
    /// The error sink receives a concrete error when the allocation fails.
    pub fn error_sink<E2>(self, error_sink: E2) -> AllocateOptions<F, E2> {
        AllocateOptions {
            fallback: self.fallback,
            error_sink,
        }
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
