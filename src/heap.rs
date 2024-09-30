//! The module of heaps and contexts.
//!
//! See [`Heap`] and [`Context`] for more information.

#[cfg(feature = "global")]
mod thread_local;

use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    hint,
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
    arena::{Arenas, Error},
    base::BaseAlloc,
    config::{SHARD_SIZE, SLAB_SIZE},
    slab::{AtomicBlockRef, BlockRef, EMPTY_SHARD, Shard, ShardList, Slab},
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
    match (size + GRANULARITY - 1) >> GRANULARITY_SHIFT {
        index @ 0..=7 => index,
        ssize => {
            debug_assert!(ssize > 0);
            // SAFETY: ssize > 0 according to the previous branch.
            let ssize = unsafe { NonZeroUsize::new_unchecked(ssize) };
            let sft = usize::BITS - ssize.leading_zeros() - 4;
            ((ssize.get() - 1) >> sft) + (sft << 3) as usize + 1
        }
    }
}

/// Gets the maximum object size of an object index.
///
/// This function is the inverse function of [`obj_size_index`].
pub const fn obj_size(index: usize) -> usize {
    (match index {
        0..=7 => index,
        i => (8 + (i & 7)) << ((i - 8) >> 3),
    }) << GRANULARITY_SHIFT
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
    /// The size type of which the sizes must be smaller than or equal to
    /// [`SMALL_MAX`](Self::SMALL_MAX).
    Small,
    /// The size type of which the sizes must lie within the range of
    /// [`SMALL_MAX`](Self::SMALL_MAX)` + 1..=`[`MEDIUM_MAX`](Self::MEDIUM_MAX).
    Medium,
    /// The size type of which the sizes must lie within the range of
    /// [`MEDIUM_MAX`](Self::MEDIUM_MAX)` + 1..=`[`LARGE_MAX`](Self::LARGE_MAX).
    Large,
    /// The size type of which the sizes must be greater than
    /// [`LARGE_MAX`](Self::LARGE_MAX)`.
    Huge,
}
use ObjSizeType::*;

impl ObjSizeType {
    /// The maximal size of small-sized objects.
    pub const SMALL_MAX: usize = 1024;
    /// The maximal size of medium-sized objects.
    pub const MEDIUM_MAX: usize = SHARD_SIZE / 2;
    /// The maximal size of larget-sized objects.
    // Should not be more than (SLAB_SIZE - SHARD_SIZE * HEADER_COUNT) / 2.
    pub const LARGE_MAX: usize = const_min((SLAB_SIZE - SHARD_SIZE) / 2, SLAB_SIZE * 15 / 32);
}

const fn const_min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
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
        let ptr: *const Self = ptr::from_ref(&self);
        ptr.addr()
    }

    fn alloc_slab(
        self: Pin<&Self>,
        count: NonZeroUsize,
        align: usize,
        direct: bool,
    ) -> Result<&'arena Shard<'arena>, Error<B>> {
        let slab = self
            .arena
            .allocate(self.thread_id(), count, align, direct)?;
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
                self.free_shards
                    .drain(|s| ptr::eq(unsafe { s.slab::<B>().0 }, &*slab));

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

impl<'arena> Bin<'arena> {
    const fn new(index: usize) -> Self {
        const fn const_max(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }

        // Bin #0 is identical to bin #1...
        let obj_size = const_max(obj_size(index), GRANULARITY);
        Bin {
            list: ShardList::DEFAULT,
            obj_size,
            min_direct_index: if index <= 1 {
                0 // ... so we here specify the same value for them.
            } else {
                direct_index(self::obj_size(index - 1)) + 1
            },
            max_direct_index: direct_index(obj_size),
        }
    }
}

/// A memory allocator unit of ferroc.
///
/// This type serves as the most direct interface exposed to users compared with
/// other intermediate structures. Users usually allocate memory from this type.
///
/// By far, only 1 heap may exist from 1 context.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Heap<'arena, 'cx, B: BaseAlloc> {
    cx: Option<Pin<&'cx Context<'arena, B>>>,
    direct_shards: [Cell<&'arena Shard<'arena>>; DIRECT_COUNT],
    shards: [Bin<'arena>; OBJ_SIZE_COUNT],
    full_shards: ShardList<'arena>,
    huge_shards: ShardList<'arena>,
    delayed_free: AtomicBlockRef<'arena>,
}

fn log_error<B: BaseAlloc>(err: Error<B>) {
    #[cfg(feature = "error-log")]
    log::error!("ferroc: {err}");
    #[cfg(not(feature = "error-log"))]
    let _ = err;
}

macro_rules! stry {
    ($e:expr) => {
        match $e {
            Ok(it) => it,
            Err(err) => {
                log_error(err);
                return None;
            }
        }
    };
}

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
            direct_shards: array![_ => Cell::new(EMPTY_SHARD.as_ref()); DIRECT_COUNT],
            shards: array![index => Bin::new(index); OBJ_SIZE_COUNT],
            full_shards: ShardList::DEFAULT,
            huge_shards: ShardList::DEFAULT,
            delayed_free: AtomicBlockRef::new(),
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
    #[cold]
    unsafe fn pop_huge(&self, size: usize, zero: bool) -> Option<NonNull<()>> {
        // SAFETY: The heap is initialized.
        let cx = unsafe { self.cx.unwrap_unchecked() };

        let count = (Slab::<B>::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
        let count = NonZeroUsize::new(count).unwrap();
        let shard = stry!(cx.alloc_slab(count, SHARD_SIZE, true));
        let delayed_free = NonNull::from(&self.delayed_free);
        stry!(shard.init_large_or_huge(size, count, delayed_free, cx.arena.base()));
        self.huge_shards.push(shard);

        // SAFETY: `shard` has free blocks.
        let block = unsafe { shard.pop_block_unchecked() };
        Some(Self::post_alloc(block, size, zero, shard))
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

    fn post_alloc(
        mut block: BlockRef<'arena>,
        size: usize,
        zero: bool,
        shard: &Shard<'arena>,
    ) -> NonNull<()> {
        let ptr = block.as_ptr();

        let ptr_slice = NonNull::from_raw_parts(ptr, size);
        track::allocate(ptr_slice, 0, zero);

        if zero {
            if shard.free_is_zero() {
                block.set_next(None);
            } else {
                unsafe { ptr_slice.as_uninit_slice_mut().fill(MaybeUninit::zeroed()) };
            }
            debug_assert!(unsafe { ptr_slice.as_ref().iter().any(|&b| b == 0) });
        }
        core::mem::forget(block);
        ptr
    }

    /// # Safety
    ///
    /// `fallback` must return an initialized heap.
    #[inline]
    unsafe fn pop<'a>(
        &'a self,
        size: usize,
        zero: bool,
        fallback: impl FnOnce() -> Option<&'a Self>,
    ) -> Option<NonNull<()>> {
        if size <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(size)
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some(block) = shard.pop_block()
        {
            return Some(Self::post_alloc(block, size, zero, shard));
        }

        let heap = if self.is_init() { self } else { fallback()? };
        debug_assert!(heap.is_init());
        // SAFETY: Heap is initialized.
        unsafe { heap.pop_contended(size, zero) }
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    #[cold]
    unsafe fn find_free_from_all(
        &self,
        bin: &Bin<'arena>,
        first_try: bool,
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
                if let Some(next) = stry!($fresh.init(
                    bin.obj_size,
                    NonNull::from(&self.delayed_free),
                    cx.arena.base()
                )) {
                    cx.free_shards.push(next);
                }
                bin.list.push($fresh);
                self.update_direct(bin);

                debug_assert!($fresh.has_free());
            };
        }

        // 1. Try to pop from the free shards;
        if bin.obj_size <= ObjSizeType::MEDIUM_MAX
            && let Some(free) = cx.free_shards.pop()
        {
            add_fresh!(free);
            return Some(free);
        }

        // 2. Try to collect & unfull some shards (freed from other threads);
        // However, full shards will be unfulled when `try_free_delayed` is called, so
        // we don't iterate and unfull it here. Keeping that list simply prevents full
        // shards being iterated everytime on the loop above.

        // 3. Try to allocate/reclaim a slab.
        let new = stry!(cx.alloc_slab(NonZeroUsize::MIN, 1, !first_try));
        add_fresh!(new);
        Some(new)
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    unsafe fn find_free(&self, bin: &Bin<'arena>, first_try: bool) -> Option<&Shard<'arena>> {
        if let Some(shard) = bin.list.current()
            && shard.collect(false)
        {
            return Some(shard);
        }

        // SAFETY: `cx` is initialized.
        unsafe { self.find_free_from_all(bin, first_try) }
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    #[cold]
    unsafe fn pop_contended(&self, size: usize, zero: bool) -> Option<NonNull<()>> {
        self.try_free_delayed(false);

        if size > ObjSizeType::LARGE_MAX {
            // SAFETY: The heap is initialized.
            return unsafe { self.pop_huge(size, zero) };
        }

        let index = obj_size_index(size);
        debug_assert!(index < OBJ_SIZE_COUNT);
        let bin = unsafe { self.shards.get_unchecked(index) };

        // SAFETY: `cx` is initialized.
        let shard = if let Some(shard) = unsafe { self.find_free(bin, true) } {
            shard
        } else {
            self.collect_cold();
            // SAFETY: `cx` is initialized.
            unsafe { self.find_free(bin, false) }?
        };

        // SAFETY: `shard` has free blocks.
        let block = unsafe { shard.pop_block_unchecked() };
        Some(Self::post_alloc(block, size, zero, shard))
    }

    /// # Safety
    ///
    /// `fallback` must return an initialized heap.
    unsafe fn pop_aligned<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> Option<&'a Self>,
    ) -> Option<NonNull<()>> {
        if layout.size() <= ObjSizeType::SMALL_MAX
            && let direct_index = direct_index(layout.size())
            // SAFETY: `direct_shards` only contains sizes that <= `SMALL_MAX`.
            && let shard = unsafe { self.direct_shards.get_unchecked(direct_index) }.get()
            && let Some(block) = shard.pop_block_aligned(layout.align())
        {
            return Some(Self::post_alloc(block, layout.size(), zero, shard));
        }

        // SAFETY: `fallback` returns an initialized heap.
        unsafe { self.pop_aligned_contended(layout, zero, fallback) }
    }

    /// # Safety
    ///
    /// `fallback` must return an initialized heap.
    #[cold]
    unsafe fn pop_aligned_contended<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> Option<&'a Self>,
    ) -> Option<NonNull<()>> {
        if layout.align() <= ObjSizeType::LARGE_MAX {
            let oversize = layout.size() + layout.align() - 1;
            // SAFETY: `fallback` returns an initialized heap.
            let overptr = unsafe { self.pop(oversize, zero, fallback) }?.cast();

            let addr = (overptr.addr().get() + layout.align() - 1) & !(layout.align() - 1);
            let ptr = overptr.with_addr(NonZeroUsize::new(addr).unwrap());
            if ptr.addr() != overptr.addr() {
                // SAFETY: `ptr` is just allocated from the same heap.
                unsafe {
                    let slab = Slab::<B>::from_ptr(ptr).unwrap_unchecked();
                    let shard = Slab::shard_meta(slab, ptr);
                    shard.as_ref().flags.set_align();
                }
                track::no_access(overptr.cast(), ptr.addr().get() - overptr.addr().get());
            }
            Some(ptr)
        } else {
            log_error::<B>(Error::Unsupported(layout));
            None
        }
    }

    /// # Safety
    ///
    /// `fallback` must return an initialized heap.
    unsafe fn allocate_inner<'a>(
        &'a self,
        layout: Layout,
        zero: bool,
        fallback: impl FnOnce() -> Option<&'a Self>,
    ) -> Option<NonNull<()>> {
        if layout.size() == 0 {
            return Some(layout.dangling().cast());
        }
        if layout.size() <= ObjSizeType::MEDIUM_MAX && layout.size() & (layout.align() - 1) == 0 {
            // SAFETY: `fallback` returns an initialized heap.
            return (unsafe { self.pop(layout.size(), zero, fallback) })
                .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())));
        }

        // SAFETY: `fallback` returns an initialized heap.
        unsafe { self.pop_aligned(layout, zero, fallback) }
            .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())))
    }

    /// Get the default allocate options of the current heap.
    ///
    /// The default heap fallback will panic, while the default error sink will
    /// silently drop the error.
    #[allow(clippy::type_complexity)]
    pub fn options<'a>() -> AllocateOptions<fn() -> Option<&'a Self>> {
        AllocateOptions { fallback: || None }
    }

    /// Allocate a memory block of `layout` with additional options.
    ///
    /// See [`AllocateOptions`] for the meaning of them.
    pub fn allocate_with<'a, F>(
        &'a self,
        layout: Layout,
        zero: bool,
        options: AllocateOptions<F>,
    ) -> Result<NonNull<()>, AllocError>
    where
        F: FnOnce() -> Option<&'a Self>,
    {
        // SAFETY: `fallback` returns an initialized heap, according to the
        // functionality of `options`.
        unsafe { self.allocate_inner(layout, zero, options.fallback) }.ok_or(AllocError)
    }

    #[cfg(feature = "c")]
    #[doc(hidden)]
    pub unsafe fn malloc<'a>(
        &'a self,
        size: usize,
        zero: bool,
        fallback: impl FnOnce() -> Option<&'a Self>,
    ) -> Option<NonNull<()>> {
        unsafe { self.pop(size, zero, fallback) }
    }

    /// Allocate a memory block of `layout`.
    ///
    /// The allocation can be deallocated by other heaps referring to the same
    /// arena collection.
    ///
    /// # Errors
    ///
    /// [`AllocError`] are returned when allocation fails.
    #[inline]
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<()>, AllocError> {
        self.allocate_with(layout, false, Self::options())
    }

    /// Allocate a zeroed memory block of `layout`.
    ///
    /// The allocation can be deallocated by other heaps referring to the same
    /// arena collection.
    ///
    /// # Errors
    ///
    /// [`AllocError`] are returned when allocation fails.
    #[inline]
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<()>, AllocError> {
        self.allocate_with(layout, true, Self::options())
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
        // SAFETY: The same as `Slab::from_ptr`.
        let shard = unsafe { Slab::shard_meta(slab, ptr.cast()) };
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
            // SAFETY: The allocation size is not 0.
            let tested_layout = unsafe { self.layout_of(ptr) };
            assert!(
                tested_layout.size() >= layout.size(),
                "the layout validation of allocation at {ptr:p} failed:\n\
                \trequest: {layout:?}\n\
                \tcalculated: {layout:?}"
            );
            assert!(
                tested_layout.align() >= layout.align(),
                "the layout validation of allocation at {ptr:p} failed:\n\
                \trequest: {layout:?}\n\
                \tcalculated: {layout:?}"
            );
        }
        // SAFETY: `ptr` is allocated by these structures.
        unsafe { self.free(ptr) }
    }

    /// # Safety
    ///
    /// `cx` must be initialized.
    unsafe fn free_shard(&self, shard: &'arena Shard<'arena>, obj_size: usize) {
        debug_assert!(shard.is_unused());
        debug_assert!(self.cx.is_some());
        // SAFETY: `cx` is initialized.
        let cx = unsafe { self.cx.unwrap_unchecked() };

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

    fn try_free_delayed(&self, no_more: bool) -> bool {
        let delayed_free = self.delayed_free.get();

        let mut ptr = delayed_free.load(Relaxed);
        loop {
            let Some(block) = NonNull::new(ptr) else {
                break true;
            };
            match delayed_free.compare_exchange_weak(ptr, ptr::null_mut(), AcqRel, Acquire) {
                Ok(_) => {
                    let block = unsafe { BlockRef::from_raw(block) };
                    break self.free_delayed_contended(block, no_more);
                }
                Err(b) => ptr = b,
            }
        }
    }

    #[cold]
    fn free_delayed_contended(&self, mut block: BlockRef<'arena>, no_more: bool) -> bool {
        let delayed_free = self.delayed_free.get();

        let mut cleared = true;
        loop {
            let next = block.take_next();

            let ptr = block.as_ptr();
            // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
            let slab = unsafe { Slab::<B>::from_ptr(ptr).unwrap_unchecked() };
            // SAFETY: The block is ours, and so is this shard.
            let shard = unsafe { Slab::shard_meta(slab, ptr.cast()).as_ref() };

            if shard.reset_delayed(no_more) {
                // SAFETY: Uninit heaps cannot have deallocated blocks.
                unsafe { self.free_block(shard, block) };
            } else {
                let mut cur = delayed_free.load(Relaxed);
                loop {
                    // SAFETY: `delayed_free` owns a list of blocks.
                    block.set_next(NonNull::new(cur).map(|p| unsafe { BlockRef::from_raw(p) }));
                    let new = block.as_ptr().as_ptr();
                    match delayed_free.compare_exchange_weak(cur, new, Release, Relaxed) {
                        Ok(_) => break,
                        Err(e) => cur = e,
                    }
                }
                cleared = false;
            }

            block = if let Some(block) = next { block } else { break }
        }
        cleared
    }

    #[inline]
    fn thread_id(&self) -> usize {
        match self.cx {
            Some(ref cx) => cx.thread_id(),
            None => 0,
        }
    }

    /// # Safety
    ///
    /// - `ptr` must point to an owned, valid memory block, previously allocated
    ///   by a certain instance of `Heap` alive in the scope, created from the
    ///   same arena.
    /// - No aliases of `ptr` should exist after the deallocation.
    /// - The allocation size must not be 0.
    #[doc(hidden)]
    pub unsafe fn free(&self, ptr: NonNull<u8>) {
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
                    // SAFETY: `cx` is initialized to have performed the allocation above.
                    unsafe { self.free_shard(shard, shard.obj_size.load(Relaxed)) }
                }
            } else {
                unsafe { self.free_contended(ptr, shard.into(), true) }
            }
        } else {
            unsafe { self.free_contended(ptr, shard, false) }
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
        track::deallocate(block.as_ptr().cast(), 0);
        if is_local {
            let shard = unsafe { shard.as_ref() };

            // SAFETY: `cx` is initialized to have performed the allocation above.
            unsafe { self.free_block(shard, block) };
        } else {
            // We're deallocating from another thread.
            unsafe { Shard::push_block_mt(shard, block) }
        }
    }

    /// # Safety
    ///
    /// `block` must be in `shard`, which currently belongs to `self`, whose
    /// `cx` must be initialized.
    unsafe fn free_block(&self, shard: &'arena Shard<'arena>, block: BlockRef<'arena>) {
        let is_unused = shard.push_block(block);
        if is_unused {
            // SAFETY: `cx` is initialized.
            unsafe { self.free_shard(shard, shard.obj_size.load(Relaxed)) }
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
        if !self.is_init() {
            return;
        }
        self.collect_inner(force);
    }

    #[cold]
    fn collect_cold(&self) {
        self.collect_inner(true);
    }

    fn collect_inner(&self, force: bool) {
        let shards = self.shards.iter().flat_map(|bin| &bin.list);
        shards.for_each(|shard| {
            shard.collect(force);
        });

        while !self.try_free_delayed(false) && force {
            hint::spin_loop();
        }

        if force && let Some(cx) = self.cx {
            cx.arena.reclaim_all();
        }
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Drop for Heap<'arena, 'cx, B> {
    fn drop(&mut self) {
        if let Some(cx) = self.cx.take() {
            self.try_free_delayed(true);

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
        self.allocate_with(layout, false, Self::options())
            .map(|t| NonNull::from_raw_parts(t, layout.size()))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate_with(layout, true, Self::options())
            .map(|t| NonNull::from_raw_parts(t, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.deallocate(ptr, layout) };
    }
}

/// The additional allocate options which can be passed into
/// [`allocate_with`](Heap::allocate_with).
#[derive(Clone, Copy)]
pub struct AllocateOptions<F> {
    fallback: F,
}

impl<F> AllocateOptions<F> {
    /// Creates a new allocate options.
    ///
    /// # Safety
    ///
    /// `fallback` must return a reference to an already initialized heap,
    /// otherwise Undefined Behavior will be raised.
    pub const unsafe fn new(fallback: F) -> Self {
        AllocateOptions { fallback }
    }

    /// Replace with a new fallback.
    ///
    /// The fallback returns lazyily evaluated backup heap when the current heap
    /// is not initialized (fails to acquire memory from its context).
    ///
    /// # Safety
    ///
    /// `fallback` must return a reference to an already initialized heap,
    /// otherwise Undefined Behavior will be raised.
    pub unsafe fn fallback<F2>(self, fallback: F2) -> AllocateOptions<F2> {
        AllocateOptions { fallback }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(miri))]
    use crate::heap::{GRANULARITY_SHIFT, obj_size, obj_size_index};

    #[test]
    #[cfg(not(miri))]
    fn test_obj_size() {
        let size_a = (0..8).map(|i| i << GRANULARITY_SHIFT);
        let size_b = |sft| (8..16usize).map(move |size| size << (sft + GRANULARITY_SHIFT));

        let mut last = 0;
        for (index, size) in size_a.chain((0..15).flat_map(size_b)).enumerate() {
            for s in last..=size {
                assert_eq!(obj_size_index(s), index);
            }
            assert_eq!(obj_size(index), size);
            last = size + 1;
        }
    }
}
