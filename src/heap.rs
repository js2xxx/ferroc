use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::{Cell, RefCell},
    num::NonZeroUsize,
    ptr::{self, NonNull},
    sync::atomic::{AtomicU64, Ordering::*},
};

use crate::{
    arena::{Arenas, Error, SHARD_SIZE, SLAB_SIZE},
    base::BaseAlloc,
    slab::{BlockRef, Shard, ShardList, Slab},
    Stat,
};

pub const OBJ_SIZES: &[usize] = &[
    16, 24, //  \ - Small
    32, 48, //  /
    64, 80, 96, 112, //    \ - Medium
    128, 160, 192, 224, // |
    256, 320, 384, 448, // |
    512, 640, 768, 896, // /
    1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, //         \ - Large
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, //         |
    4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, //         |
    8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, //   |
    16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, // |
    32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, // /
    65536, // SHARD_SIZE_MAX
];

pub(crate) const OBJ_SIZE_COUNT: usize = OBJ_SIZES.len();

pub fn obj_size_index_tabled(size: usize) -> Option<usize> {
    let (Ok(index) | Err(index)) = OBJ_SIZES.binary_search(&size);
    match OBJ_SIZES.get(index) {
        Some(&size) if size <= SHARD_SIZE => Some(index),
        _ => None,
    }
}

pub const fn obj_size_index(size: usize) -> Option<usize> {
    assert!(size != 0);
    if size > SHARD_SIZE {
        return None;
    }
    let size_m1 = size - 1;
    let msb = match (usize::BITS - size_m1.leading_zeros()).checked_sub(1) {
        Some(msb) => msb as usize,
        None => return Some(0), // size = 1.
    };
    let index = match msb {
        0..=3 => return Some(0),
        4..=5 => ((msb - 4) << 1) + ((size_m1 >> (msb - 1)) & 1),
        6..=9 => ((msb - 6) << 2) + ((size_m1 >> (msb - 2)) & 3) + 4,
        10..=17 => ((msb - 10) << 3) + ((size_m1 >> (msb - 3)) & 7) + 4 + 16,
        18..=33 => ((msb - 18) << 4) + ((size_m1 >> (msb - 4)) & 15) + 4 + 16 + 64,
        34..=65 => ((msb - 34) << 5) + ((size_m1 >> (msb - 5)) & 31) + 4 + 16 + 64 + 256,
        _ => return None, // There's no machine of `usize` greater than 64-bit by far.
    };
    Some(index + 1)
}

pub struct Context<'arena, B: BaseAlloc> {
    thread_id: u64,
    arena: &'arena Arenas<B>,
    free_shards: ShardList<'arena>,
    heap_count: Cell<usize>,
    stat: RefCell<Stat>,
}

impl<'arena, B: BaseAlloc> Context<'arena, B> {
    pub fn new(arena: &'arena Arenas<B>) -> Self {
        static ID: AtomicU64 = AtomicU64::new(1);
        Context {
            thread_id: ID.fetch_add(1, Relaxed),
            arena,
            free_shards: Default::default(),
            heap_count: Cell::new(0),
            #[cfg(feature = "stat")]
            stat: RefCell::new(Stat::INIT),
            #[cfg(not(feature = "stat"))]
            stat: RefCell::new(()),
        }
    }

    fn alloc_slab(
        &self,
        count: NonZeroUsize,
        align: usize,
        is_huge: bool,
        _stat: &mut Stat,
    ) -> Result<&'arena Shard<'arena>, Error<B>> {
        let slab = self.arena.allocate(self.thread_id, count, align, is_huge)?;
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

pub struct Heap<'arena: 'cx, 'cx, B: BaseAlloc> {
    cx: &'cx Context<'arena, B>,
    shards: [ShardList<'arena>; OBJ_SIZE_COUNT],
    full_shards: ShardList<'arena>,
    huge_shards: ShardList<'arena>,
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Heap<'arena, 'cx, B> {
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

    fn pop_huge(&self, size: usize, stat: &mut Stat) -> Result<NonNull<[u8]>, Error<B>> {
        debug_assert!(size > SHARD_SIZE);

        let count = (Slab::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
        let count = NonZeroUsize::new(count).unwrap();
        let shard = self.cx.alloc_slab(count, SLAB_SIZE, true, stat)?;
        shard.slab().0.inc_used();
        shard.init_huge(size, stat);
        self.huge_shards.push(shard);

        let block = shard.pop_block().unwrap();
        #[cfg(feature = "stat")]
        {
            stat.huge_count += 1;
            stat.huge_size += size;
        }
        Ok(NonNull::from_raw_parts(block.into_raw(), size))
    }

    fn pop(&self, size: usize, stat: &mut Stat) -> Result<NonNull<[u8]>, Error<B>> {
        let index = match obj_size_index(size) {
            Some(index) => index,
            None => return self.pop_huge(size, stat),
        };

        let block = if let Some(shard) = self.shards[index].current()
            && let Some(block) = shard.pop_block()
        {
            #[cfg(feature = "stat")]
            {
                stat.normal_count[index] += 1;
                stat.normal_size += OBJ_SIZES[index];
            }
            block
        } else {
            self.pop_contended(index, stat)?
        };

        Ok(NonNull::from_raw_parts(block.into_raw(), size))
    }

    #[cold]
    fn pop_contended(&self, index: usize, stat: &mut Stat) -> Result<BlockRef<'arena>, Error<B>> {
        let list = &self.shards[index];

        let pop_from_list = |_stat: &mut Stat| {
            let mut cursor = list.cursor_head();
            loop {
                let shard = *cursor.get()?;
                shard.collect(false);
                shard.extend();

                match shard.pop_block() {
                    Some(block) => {
                        #[cfg(feature = "stat")]
                        {
                            _stat.normal_count[index] += 1;
                            _stat.normal_size += OBJ_SIZES[index];
                        }
                        break Some(block);
                    }
                    None => {
                        cursor.remove();
                        shard.is_in_full.set(true);
                        self.full_shards.push(shard)
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
            if let Some(next) = free.init(OBJ_SIZES[index], _stat) {
                self.cx.free_shards.push(next);
                #[cfg(feature = "stat")]
                {
                    _stat.free_shards += next.shard_count();
                }
            }
            list.push(free);
        };

        if !list.is_empty()
            && let Some(block) = pop_from_list(stat)
        {
            return Ok(block);
        }

        if let Some(free) = self.cx.free_shards.pop() {
            // 1. Try to pop from the free shards;
            add_free(free, stat);
        } else {
            // 2. Try to collect potentially unfull shards.
            let unfulled = self.full_shards.drain(|shard| {
                shard.collect(false);
                !shard.is_full()
            });
            let mut has_unfulled = false;
            unfulled.for_each(|shard| {
                let i = obj_size_index(shard.obj_size.load(Relaxed)).unwrap();
                self.shards[i].push(shard);
                has_unfulled |= i == index;
            });

            // 3. Try to clear abandoned huge shards and allocate/reclaim a slab.
            if !has_unfulled {
                self.collect_huge(stat);
                let free = self
                    .cx
                    .alloc_slab(NonZeroUsize::MIN, SLAB_SIZE, false, stat)?;
                add_free(free, stat);
            }
        }

        Ok(pop_from_list(stat).unwrap())
    }

    fn pop_aligned(&self, layout: Layout, stat: &mut Stat) -> Result<NonNull<[u8]>, Error<B>> {
        let ptr = match obj_size_index(layout.size()) {
            Some(index)
                if let Some(shard) = self.shards[index].current()
                    && let Some(block) = shard.pop_block_aligned(layout.align()) =>
            {
                #[cfg(feature = "stat")]
                {
                    stat.normal_count[index] += 1;
                    stat.normal_size += OBJ_SIZES[index];
                }
                block.into_raw()
            }
            None if layout.align() <= SHARD_SIZE => return self.pop_huge(layout.size(), stat),
            _ if layout.align() < SLAB_SIZE => {
                let ptr = self.pop(layout.size() + layout.align() - 1, stat)?;
                ptr.cast().map_addr(|addr| unsafe {
                    NonZeroUsize::new_unchecked(
                        (addr.get() + layout.align() - 1) & !(layout.align() - 1),
                    )
                })
            }
            _ => return self.cx.arena.allocate_direct(layout),
        };
        Ok(NonNull::from_raw_parts(ptr, layout.size()))
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<B>> {
        if layout.size() == 0 {
            // SAFETY: Alignments are not zero.
            return Ok(unsafe {
                let addr = NonNull::new_unchecked(ptr::invalid_mut(layout.align()));
                NonNull::from_raw_parts(addr, 0)
            });
        }
        let mut stat = self.cx.stat.borrow_mut();
        if layout.size() <= SHARD_SIZE && layout.size() % layout.align() == 0 {
            return self.pop(layout.size(), &mut stat);
        }
        self.pop_aligned(layout, &mut stat)
    }

    pub fn stat(&self) -> Stat {
        *self.cx.stat.borrow()
    }

    /// # Safety
    ///
    /// `ptr` must point to an owned, valid memory block of `layout`, previously
    /// allocated by a certain instance of `Heap` alive in the scope, created
    /// from the same arena.
    pub unsafe fn layout_of(&self, ptr: NonNull<u8>) -> Option<Layout> {
        if ptr.is_aligned_to(SLAB_SIZE) {
            return self.cx.arena.layout_of_direct(ptr);
        }
        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let Some(slab) = (unsafe { Slab::from_ptr(ptr) }) else {
            return Layout::from_size_align(0, ptr.addr().get()).ok();
        };
        // SAFETY: `ptr` is in `slab`.
        let (_, block, obj_size) = unsafe { Slab::shard_infos(slab, ptr.cast()) };
        let size = obj_size - (block.into_raw().addr().get() - ptr.addr().get());
        let align = 1 << ptr.addr().get().trailing_zeros();
        Some(Layout::from_size_align(size, align).unwrap())
    }

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

        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let Some(slab) = (unsafe { Slab::from_ptr(ptr) }) else {
            return;
        };
        let thread_id = unsafe { ptr::addr_of!((*slab.as_ptr()).thread_id).read() };
        // SAFETY: `ptr` is in `slab`.
        let (shard, block, obj_size) = unsafe { Slab::shard_infos(slab, ptr.cast()) };
        if self.cx.thread_id == thread_id {
            // `thread_id` matches; We're deallocating from the same thread.
            let _stat = &mut *self.cx.stat.borrow_mut();

            let shard = unsafe { shard.as_ref() };
            let was_full = shard.is_in_full.replace(false);

            let is_unused = shard.push_block(block);

            if let Some(index) = obj_size_index(obj_size) {
                #[cfg(feature = "stat")]
                {
                    _stat.normal_count[index] -= 1;
                    _stat.normal_size -= OBJ_SIZES[index];
                }

                if was_full {
                    self.full_shards.remove(shard);
                    self.shards[index].push(shard);
                }

                if is_unused && self.shards[index].len() > 1 {
                    self.shards[index].remove(shard);
                    self.cx.finalize_shard(shard, _stat);
                }
            } else {
                #[cfg(feature = "stat")]
                {
                    _stat.huge_count -= 1;
                    _stat.huge_size -= obj_size;
                }
                debug_assert!(is_unused);

                self.huge_shards.remove(shard);
                self.cx.finalize_shard(shard, _stat);
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

    pub fn collect(&self, force: bool) {
        let shards = self.shards.iter().flatten();
        shards.for_each(|shard| shard.collect(force));

        self.collect_huge(&mut self.cx.stat.borrow_mut());
    }
}

impl<'arena: 'cx, 'cx, B: BaseAlloc> Drop for Heap<'arena, 'cx, B> {
    fn drop(&mut self) {
        let mut stat = self.cx.stat.borrow_mut();
        let iter = (self.shards.iter()).chain([&self.huge_shards, &self.full_shards]);
        iter.flat_map(|l| l.drain(|_| true)).for_each(|shard| {
            shard.collect(false);
            self.cx.finalize_shard(shard, &mut stat);
        });
        self.cx.heap_count.set(self.cx.heap_count.get() - 1);
    }
}

unsafe impl<'arena: 'cx, 'cx, B: BaseAlloc> Allocator for Heap<'arena, 'cx, B> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout).map_err(|_| AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arena::SHARD_SIZE,
        heap::{obj_size_index, obj_size_index_tabled},
    };

    #[test]
    fn test_osi() {
        for s in 1..=SHARD_SIZE {
            let calculated = obj_size_index_tabled(s);
            let table = obj_size_index(s);
            assert_eq!(
                calculated, table,
                "size {s}: calculated = {calculated:?}, table = {table:?}"
            );
        }
    }
}
