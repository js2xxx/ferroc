use core::{
    alloc::{AllocError, Allocator, Layout},
    num::NonZeroUsize,
    ptr::{self, NonNull},
    sync::atomic::{AtomicU64, Ordering::*},
};

use crate::{
    arena::{Arenas, Error, SHARD_SIZE, SLAB_SIZE},
    os::OsAlloc,
    slab::{BlockRef, Shard, ShardList, Slab, SlabRef},
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
    32768, 36864, 40960, 45056, 49144, 53248, 57344, 61440, // /
    65536, // SHARD_SIZE_MAX
];

pub const OBJ_SIZE_COUNT: usize = OBJ_SIZES.len();

pub fn obj_size_index(size: usize) -> Option<usize> {
    Some(match OBJ_SIZES.binary_search(&size) {
        Ok(index) => index,
        Err(index) => match OBJ_SIZES.get(index) {
            Some(&size) if size <= SHARD_SIZE => index,
            _ => return None,
        },
    })
}

pub struct Context<'a, Os: OsAlloc> {
    thread_id: u64,
    arena: &'a Arenas<Os>,
    free_shards: ShardList<'a>,
    abandoned_shards: ShardList<'a>,
}

impl<'a, Os: OsAlloc> Context<'a, Os> {
    pub fn new(arena: &'a Arenas<Os>) -> Self {
        static ID: AtomicU64 = AtomicU64::new(0);
        Context {
            thread_id: ID.fetch_add(1, Relaxed),
            arena,
            free_shards: Default::default(),
            abandoned_shards: Default::default(),
        }
    }

    fn alloc_slab(
        &self,
        count: NonZeroUsize,
        align: usize,
        is_huge: bool,
    ) -> Result<&'a Shard<'a>, Error<Os>> {
        let slab = self.arena.allocate(self.thread_id, count, align, is_huge)?;
        Ok(slab.into_shard())
    }

    /// # Safety
    ///
    /// No more references to the `slab` or its shards should exist after
    /// calling this function.
    unsafe fn dealloc_slab(&self, slab: SlabRef<'a>) {
        unsafe { self.arena.deallocate(slab) }
    }

    /// # Safety
    ///
    /// No more references to the `slab` or its shards should exist after
    /// calling this function.
    unsafe fn finalize_shard(&self, shard: &'a Shard<'a>) {
        match shard.fini() {
            Ok(fini) => self.free_shards.push(fini),
            // `slab` is unused, we can deallocate it.
            Err(slab) => unsafe { self.dealloc_slab(slab) },
        }
    }

    pub fn collect(&self, force: bool) {
        self.abandoned_shards
            .iter()
            .for_each(|shard| shard.collect(force));
    }
}

pub struct Heap<'a, Os: OsAlloc> {
    cx: &'a Context<'a, Os>,
    shards: [ShardList<'a>; OBJ_SIZE_COUNT],
    full_shards: ShardList<'a>,
    huge_shards: ShardList<'a>,
}

impl<'a, Os: OsAlloc> Heap<'a, Os> {
    pub fn new(cx: &'a Context<'a, Os>) -> Self {
        Heap {
            cx,
            shards: [ShardList::DEFAULT; OBJ_SIZE_COUNT],
            full_shards: ShardList::DEFAULT,
            huge_shards: ShardList::DEFAULT,
        }
    }

    fn pop_huge(&self, size: usize) -> Result<NonNull<[u8]>, Error<Os>> {
        debug_assert!(size > SHARD_SIZE);

        let count = (Slab::HEADER_COUNT * SHARD_SIZE + size).div_ceil(SLAB_SIZE);
        let count = NonZeroUsize::new(count).unwrap();
        let shard = self.cx.alloc_slab(count, SLAB_SIZE, true)?;
        shard.init_huge(size);
        self.huge_shards.push(shard);

        let block = shard.pop_block().unwrap();
        Ok(NonNull::from_raw_parts(block.into_raw(), size))
    }

    fn pop(&self, size: usize) -> Result<NonNull<[u8]>, Error<Os>> {
        let index = match obj_size_index(size) {
            Some(index) => index,
            None => return self.pop_huge(size),
        };

        let block = if let Some(shard) = self.shards[index].current()
            && let Some(block) = shard.pop_block()
        {
            block
        } else {
            self.pop_contended(index)?
        };

        Ok(NonNull::from_raw_parts(block.into_raw(), size))
    }

    #[cold]
    fn pop_contended(&self, index: usize) -> Result<BlockRef<'a>, Error<Os>> {
        let list = &self.shards[index];

        let pop_from_list = || {
            let mut cursor = list.cursor_head();
            loop {
                let shard = *cursor.get()?;
                shard.collect(false);
                shard.extend();

                match shard.pop_block() {
                    Some(block) => break Some(block),
                    None => {
                        cursor.remove();
                        shard.is_in_full.set(true);
                        self.full_shards.push(shard)
                    }
                }
            }
        };

        if !list.is_empty()
            && let Some(block) = pop_from_list()
        {
            return Ok(block);
        }

        if let Some(free) = (self.cx.free_shards.pop()) // 1. Try to pop from the free shards;
            .or_else(|| {
                // 2. Try to collect abandoned shards (only has `free` & `thread_free` blocks);
                let shard = self.cx.abandoned_shards.iter().find(|shard| {
                    shard.collect(false);
                    shard.is_unused()
                })?;
                self.cx.abandoned_shards.remove(shard);
                Some(shard)
            })
        {
            if let Some(next) = free.init(OBJ_SIZES[index]) {
                self.cx.free_shards.push(next);
            }
            list.push(free);
        } else {
            // 3. Try to collect potentially unfull shards.
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

            // 4. Try to clear abandoned huge shards and allocate a new slab.
            if !has_unfulled {
                self.clear_abandoned_huge();
                let free = self.cx.alloc_slab(NonZeroUsize::MIN, SLAB_SIZE, false)?;
                if let Some(next) = free.init(OBJ_SIZES[index]) {
                    self.cx.free_shards.push(next);
                }
                list.push(free);
            }
        }

        Ok(pop_from_list().unwrap())
    }

    fn pop_aligned(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<Os>> {
        let ptr = match obj_size_index(layout.size()) {
            Some(index)
                if let Some(shard) = self.shards[index].current()
                    && let Some(block) = shard.pop_block_aligned(layout.align()) =>
            {
                block.into_raw()
            }
            None if layout.align() <= SHARD_SIZE => return self.pop_huge(layout.size()),
            _ if layout.align() < SLAB_SIZE => {
                let ptr = self.pop(layout.size() + layout.align() - 1)?;
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

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, Error<Os>> {
        if layout.size() <= SHARD_SIZE && layout.size() % layout.align() == 0 {
            return self.pop(layout.size());
        }
        self.pop_aligned(layout)
    }

    /// # Safety
    ///
    /// `ptr` must point to an owned, valid memory block of `layout`, previously
    /// allocated by a certain instance of `Heap` alive in the scope.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.align() >= SLAB_SIZE {
            unsafe { self.cx.arena.deallocate_direct(ptr, layout) };
            return;
        }

        // SAFETY: We don't obtain the actual reference of it, as slabs aren't `Sync`.
        let slab = unsafe { Slab::from_ptr(ptr) };
        let thread_id = unsafe { ptr::addr_of!((*slab.as_ptr()).thread_id).read() };
        // SAFETY: `ptr` is in `slab`.
        let (shard, block, obj_size) = unsafe { Slab::shard_infos(slab, ptr.cast(), layout) };
        if self.cx.thread_id == thread_id {
            // `thread_id` matches; We're deallocating from the same thread.
            let shard = unsafe { shard.as_ref() };
            let was_full = shard.is_in_full.replace(false);
            let is_unused = shard.push_block(block);

            if let Some(index) = obj_size_index(obj_size) {
                if was_full {
                    self.full_shards.remove(shard);
                    self.shards[index].push(shard);
                }

                if is_unused && self.shards[index].len() > 1 {
                    self.shards[index].remove(shard);
                    // `shard` is unused after this calling.
                    unsafe { self.cx.finalize_shard(shard) }
                }
            } else {
                debug_assert!(is_unused);

                self.huge_shards.remove(shard);
                // `shard` is unused after this calling.
                unsafe { self.cx.finalize_shard(shard) }
            }
        } else {
            // We're deallocating from another thread.
            unsafe { Shard::push_block_mt(shard, block) }
        }
    }

    fn clear_abandoned_huge(&self) -> bool {
        let huge = self.huge_shards.drain(|shard| {
            shard.collect(false);
            shard.is_unused()
        });
        let mut has_unused = false;
        huge.for_each(|shard| {
            // `shard` is unused after this calling.
            unsafe { self.cx.finalize_shard(shard) };
            has_unused = true
        });
        has_unused
    }

    pub fn collect(&self, force: bool) {
        self.cx.collect(force);

        let shards = self.shards.iter().flatten();
        shards.for_each(|shard| shard.collect(force));

        self.clear_abandoned_huge();
    }
}

impl<'a, Os: OsAlloc> Drop for Heap<'a, Os> {
    fn drop(&mut self) {
        let iter = self.shards.iter().flat_map(|l| l.drain(|_| true));
        iter.for_each(|shard| {
            shard.collect(false);
            if shard.is_unused() {
                // `shard` is unused after this calling.
                unsafe { self.cx.finalize_shard(shard) }
            } else {
                self.cx.abandoned_shards.push(shard)
            }
        });
        self.clear_abandoned_huge();
    }
}

unsafe impl<'a, Os: OsAlloc> Allocator for Heap<'a, Os> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout).map_err(|_| AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}
