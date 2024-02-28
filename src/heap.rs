//! The module of heaps and contexts.
//!
//! See [`Heap`] and [`Context`] for more information.

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

use crate::{
    arena::{Arenas, Error, SHARD_SIZE, SLAB_SIZE},
    base::BaseAlloc,
    slab::{BlockRef, Shard, ShardList, Slab},
    track, Stat,
};

/// The count of small & medium object sizes.
pub const OBJ_SIZE_COUNT: usize = obj_size_index(ObjSizeType::LARGE_MAX) + 1;

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
use ObjSizeType::*;

impl ObjSizeType {
    pub const SMALL_MAX: usize = 64;
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
        Context {
            thread_id: ID.fetch_add(1, Relaxed),
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

/// A memory allocator unit of ferroc.
///
/// This type serves as the most direct interface exposed to users compared with
/// other intermediate structures. Users usually allocate memory from this type.
///
/// By far, only 1 heap may exist from 1 context.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Heap<'arena: 'cx, 'cx, B: BaseAlloc> {
    cx: &'cx Context<'arena, B>,
    shards: [ShardList<'arena>; OBJ_SIZE_COUNT],
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
    pub fn new(cx: &'cx Context<'arena, B>) -> Self {
        const MAX_HEAP_COUNT: usize = 1;

        let count = cx.heap_count.get() + 1;
        assert!(
            count <= MAX_HEAP_COUNT,
            "a context can only have at most {MAX_HEAP_COUNT} heap(s)"
        );
        cx.heap_count.set(count);
        Heap {
            cx,
            shards: [ShardList::DEFAULT; OBJ_SIZE_COUNT],
            full_shards: ShardList::DEFAULT,
            huge_shards: ShardList::DEFAULT,
        }
    }

    fn pop_huge_untracked(
        &self,
        size: usize,
        stat: &mut Stat,
        set_align: bool,
    ) -> Result<(NonNull<[u8]>, bool), Error<B>> {
        debug_assert!(size > ObjSizeType::LARGE_MAX);

        let count = (Slab::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
        let count = NonZeroUsize::new(count).unwrap();
        let shard = self.cx.alloc_slab(count, SLAB_SIZE, true, stat)?;
        shard.slab().0.inc_used();
        shard.init_large_or_huge(size, count, self.cx.arena.base(), stat)?;
        self.huge_shards.push(shard);

        let (block, is_zeroed) = shard.pop_block().unwrap();
        #[cfg(feature = "stat")]
        {
            stat.huge_count += 1;
            stat.huge_size += size;
        }
        if set_align {
            shard.has_aligned.store(true, Relaxed);
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

    fn pop_untracked(
        &self,
        size: usize,
        stat: &mut Stat,
        set_align: bool,
    ) -> Result<(NonNull<[u8]>, bool), Error<B>> {
        let ty = match obj_size_type(size) {
            ty @ (Small | Medium | Large) => ty,
            Huge => return self.pop_huge_untracked(size, stat, set_align),
        };
        let index = obj_size_index(size);

        let (block, is_zeroed) = if let Some(shard) = self.shards[index].current()
            && let Some(ret) = shard.pop_block()
        {
            #[cfg(feature = "stat")]
            {
                stat.normal_count[index] += 1;
                stat.normal_size += obj_size(index);
            }
            if set_align {
                shard.has_aligned.store(true, Relaxed);
            }
            ret
        } else {
            self.pop_contended(index, stat, ty, set_align)?
        };

        Ok((NonNull::from_raw_parts(block.into_raw(), size), is_zeroed))
    }

    #[inline]
    fn pop(&self, size: usize, zero: bool, stat: &mut Stat) -> Result<NonNull<[u8]>, Error<B>> {
        let (ptr, is_zeroed) = self.pop_untracked(size, stat, false)?;
        post_alloc(ptr, is_zeroed, zero);
        Ok(ptr)
    }

    #[cold]
    fn pop_contended(
        &self,
        index: usize,
        stat: &mut Stat,
        ty: ObjSizeType,
        set_align: bool,
    ) -> Result<(BlockRef<'arena>, bool), Error<B>> {
        let list = &self.shards[index];

        let pop_from_list = |_stat: &mut Stat| {
            let mut cursor = list.cursor_head();
            loop {
                let shard = cursor.get()?;
                shard.collect(false);
                shard.extend();

                match shard.pop_block() {
                    Some(block) => {
                        #[cfg(feature = "stat")]
                        {
                            _stat.normal_count[index] += 1;
                            _stat.normal_size += obj_size(index);
                        }
                        if set_align {
                            shard.has_aligned.store(true, Relaxed);
                        }
                        break Some(block);
                    }
                    None => {
                        cursor.remove();
                        shard.is_in_full.set(true);
                        self.full_shards.push(shard);
                    }
                }
            }
        };

        let add_free = |free: &'arena Shard<'arena>, _stat: &mut Stat| {
            #[cfg(feature = "stat")]
            {
                _stat.free_shards -= free.shard_count();
            }
            free.slab().0.inc_used();
            if let Some(next) = free.init(obj_size(index), self.cx.arena.base(), _stat)? {
                self.cx.free_shards.push(next);
                #[cfg(feature = "stat")]
                {
                    _stat.free_shards += next.shard_count();
                }
            }
            list.push(free);
            Ok(())
        };

        if !list.is_empty()
            && let Some(block) = pop_from_list(stat)
        {
            return Ok(block);
        }

        let is_large = matches!(ty, Large);
        if !is_large && let Some(free) = self.cx.free_shards.pop() {
            // 1. Try to pop from the free shards;
            add_free(free, stat)?;
        } else {
            // 2. Try to collect potentially unfull shards.
            let unfulled = self.full_shards.drain(|shard| {
                shard.collect(false);
                !shard.is_full()
            });
            let mut has_unfulled = false;
            unfulled.for_each(|shard| {
                let i = obj_size_index(shard.obj_size.load(Relaxed));
                shard.is_in_full.set(false);
                self.shards[i].push(shard);
                has_unfulled |= i == index;
            });

            // 3. Try to clear abandoned huge shards and allocate/reclaim a slab.
            if !has_unfulled {
                self.collect_huge(stat);
                let free = self
                    .cx
                    .alloc_slab(NonZeroUsize::MIN, SLAB_SIZE, is_large, stat)?;
                add_free(free, stat)?;
            }
        }

        Ok(pop_from_list(stat).unwrap())
    }

    fn pop_aligned(
        &self,
        layout: Layout,
        zero: bool,
        stat: &mut Stat,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        match obj_size_type(layout.size()) {
            Small | Medium | Large
                if let index = obj_size_index(layout.size())
                    && let Some(shard) = self.shards[index].current()
                    && let Some((block, is_zeroed)) = shard.pop_block_aligned(layout.align()) =>
            {
                #[cfg(feature = "stat")]
                {
                    stat.normal_count[index] += 1;
                    stat.normal_size += obj_size(index);
                }
                let ptr = NonNull::from_raw_parts(block.into_raw(), layout.size());
                post_alloc(ptr, is_zeroed, zero);
                Ok(ptr)
            }
            Huge if layout.align() <= SHARD_SIZE => self.pop_huge(layout.size(), zero, stat),
            _ => self.pop_aligned_contended(layout, zero, stat),
        }
    }

    #[cold]
    fn pop_aligned_contended(
        &self,
        layout: Layout,
        zero: bool,
        stat: &mut Stat,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.align() <= SHARD_SIZE {
            let (ptr, is_zeroed) =
                self.pop_untracked(layout.size() + layout.align() - 1, stat, true)?;
            let addr = (ptr.addr().get() + layout.align() - 1) & !(layout.align() - 1);
            let ptr = ptr.cast().with_addr(NonZeroUsize::new(addr).unwrap());

            let ptr = NonNull::from_raw_parts(ptr, layout.size());
            post_alloc(ptr, is_zeroed, zero);
            Ok(ptr)
        } else {
            self.cx.arena.allocate_direct(layout, zero)
        }
    }

    #[doc(hidden)]
    pub unsafe fn allocate_inner(
        &self,
        layout: Layout,
        zero: bool,
    ) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.size() == 0 {
            // SAFETY: Alignments are not zero.
            return Ok(unsafe {
                let ptr = ptr::without_provenance_mut(layout.align());
                NonNull::from_raw_parts(NonNull::new_unchecked(ptr), 0)
            });
        }
        #[cfg(feature = "stat")]
        let mut stat = self.cx.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();
        if layout.size() & (layout.align() - 1) == 0 {
            return (self.pop(layout.size(), zero, &mut stat))
                .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())));
        }
        self.pop_aligned(layout, zero, &mut stat)
            .inspect(|p| debug_assert!(p.is_aligned_to(layout.align())))
    }

    #[cfg(feature = "c")]
    pub(crate) fn malloc(&self, size: NonZeroUsize, zero: bool) -> Result<NonNull<[u8]>, Error<B>> {
        #[cfg(feature = "stat")]
        let mut stat = self.cx.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();
        self.pop(size.get(), zero, &mut stat)
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
        *self.cx.stat.borrow()
    }

    /// Retrieves the layout information of a specific allocation.
    ///
    /// The layout returned may not be the same of the layout passed to
    /// [`allocate`](Heap::allocate), but is the most fit layout of it, and can
    /// be passed to [`deallocate`](Heap::deallocate).
    ///
    /// # Safety
    ///
    /// `ptr` must point to an owned, valid memory block of `layout`, previously
    /// allocated by a certain instance of `Heap` alive in the scope, created
    /// from the same arena.
    pub unsafe fn layout_of(&self, ptr: NonNull<u8>) -> Option<Layout> {
        #[cfg(debug_assertions)]
        if !self.cx.arena.check_ptr(ptr) {
            return None;
        }
        if ptr.is_aligned_to(SLAB_SIZE) {
            return self.cx.arena.layout_of_direct(ptr);
        }
        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let Some(slab) = (unsafe { Slab::from_ptr(ptr) }) else {
            return Layout::from_size_align(0, ptr.addr().get()).ok();
        };
        // SAFETY: `ptr` is in `slab`.
        let (shard, block) = unsafe { Slab::shard_infos(slab, ptr.cast()) };
        let obj_size = unsafe { Shard::obj_size_raw(shard) };
        let size = obj_size - (block.into_raw().addr().get() - ptr.addr().get());
        let align = 1 << ptr.addr().get().trailing_zeros();
        Some(Layout::from_size_align(size, align).unwrap())
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
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        #[cfg(debug_assertions)]
        {
            let tested_layout = self
                .layout_of(ptr)
                .expect("`ptr` is not allocated from these arenas");
            debug_assert!(tested_layout.size() >= _layout.size());
            debug_assert!(tested_layout.align() >= _layout.align());
        }
        // SAFETY: `ptr` is allocated by these structures.
        unsafe { self.free(ptr) }
    }

    /// # Safety
    ///
    /// - `ptr` must point to an owned, valid memory block, previously allocated
    ///   by a certain instance of `Heap` alive in the scope, created from the
    ///   same arena.
    /// - No aliases of `ptr` should exist after the deallocation.
    #[inline]
    pub(crate) unsafe fn free(&self, ptr: NonNull<u8>) {
        if ptr.is_aligned_to(SLAB_SIZE) {
            unsafe { self.cx.arena.deallocate_direct(ptr) };
            return;
        }
        #[cfg(debug_assertions)]
        if !self.cx.arena.check_ptr(ptr) {
            // panic!("{ptr:p} is not allocated from these arenas");
            return;
        }
        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let Some(slab) = (unsafe { Slab::from_ptr(ptr) }) else {
            return;
        };
        track::deallocate(ptr, 0);

        let thread_id = unsafe { ptr::addr_of!((*slab.as_ptr()).thread_id).read() };
        // SAFETY: `ptr` is in `slab`.
        let (shard, block) = unsafe { Slab::shard_infos(slab, ptr.cast()) };
        if self.cx.thread_id == thread_id {
            // `thread_id` matches; We're deallocating from the same thread.
            #[cfg(feature = "stat")]
            let mut stat = self.cx.stat.borrow_mut();
            #[cfg(not(feature = "stat"))]
            let mut stat = ();

            let shard = unsafe { shard.as_ref() };

            let was_full = shard.is_in_full.replace(false);
            let is_unused = shard.push_block(block);

            if is_unused || was_full {
                let obj_size = shard.obj_size.load(Relaxed);
                if let Small | Medium | Large = obj_size_type(obj_size) {
                    let list = &self.shards[obj_size_index(obj_size)];
                    #[cfg(feature = "stat")]
                    {
                        stat.normal_count[index] -= 1;
                        stat.normal_size -= self::obj_size(index);
                    }

                    if was_full {
                        let _ret = self.full_shards.remove(shard);
                        debug_assert!(_ret);
                        list.push(shard);
                    }

                    if is_unused && list.len() > 1 {
                        let _ret = list.remove(shard);
                        debug_assert!(_ret);
                        self.cx.finalize_shard(shard, &mut stat);
                    }
                } else {
                    #[cfg(feature = "stat")]
                    {
                        stat.huge_count -= 1;
                        stat.huge_size -= obj_size;
                    }

                    self.huge_shards.remove(shard);
                    self.cx.finalize_shard(shard, &mut stat);
                }
            }
        } else {
            // We're deallocating from another thread.
            unsafe { Shard::push_block_mt(shard, block) }
        }
    }

    fn collect_huge(&self, stat: &mut Stat) {
        let huge = self.huge_shards.drain(|shard| {
            shard.collect(false);
            shard.is_unused()
        });
        huge.for_each(|shard| self.cx.finalize_shard(shard, stat));
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
        let shards = self.shards.iter().flatten();
        shards.for_each(|shard| shard.collect(force));

        #[cfg(feature = "stat")]
        let mut stat = self.cx.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();

        self.collect_huge(&mut stat);
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Drop for Heap<'arena, 'cx, B> {
    fn drop(&mut self) {
        #[cfg(feature = "stat")]
        let mut stat = self.cx.stat.borrow_mut();
        #[cfg(not(feature = "stat"))]
        let mut stat = ();

        let iter = (self.shards.iter()).chain([&self.huge_shards, &self.full_shards]);
        iter.flat_map(|l| l.drain(|_| true)).for_each(|shard| {
            shard.collect(false);
            shard.is_in_full.set(false);
            self.cx.finalize_shard(shard, &mut stat);
        });
        self.cx.heap_count.set(self.cx.heap_count.get() - 1);
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
        self.deallocate(ptr, layout)
    }
}

#[cfg(test)]
mod tests {
    use crate::heap::{obj_size, obj_size_index};

    #[test]
    fn test_obj_size() {
        assert_eq!(obj_size_index(16), 0);
        assert_eq!(obj_size_index(32), 1);
        assert_eq!(obj_size_index(48), 2);
        assert_eq!(obj_size_index(64), 3);
        assert_eq!(obj_size_index(80), 4);
        assert_eq!(obj_size_index(96), 5);
        assert_eq!(obj_size_index(112), 6);
        assert_eq!(16, obj_size(0));
        assert_eq!(32, obj_size(1));
        assert_eq!(48, obj_size(2));
        assert_eq!(64, obj_size(3));
        assert_eq!(80, obj_size(4));
        assert_eq!(96, obj_size(5));
        assert_eq!(112, obj_size(6));
    }
}
