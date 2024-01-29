mod cell_link;

use core::{
    alloc::Layout,
    cell::Cell,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    ptr::{self, addr_of_mut, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering::*},
};

use self::cell_link::{CellLink, CellLinked, CellList};
use crate::arena::{SHARD_COUNT, SHARD_SIZE, SLAB_SIZE};

/// A big, sharded chunk of memory.
///
/// The first (few) shards are reserved for the `Slab` header.
#[repr(transparent)]
pub(crate) struct SlabRef<'a>(NonNull<()>, PhantomData<&'a ()>);

impl<'a> Deref for SlabRef<'a> {
    type Target = Slab<'a>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The slab contains valid slab data.
        unsafe { self.0.cast().as_ref() }
    }
}

impl<'a> SlabRef<'a> {
    pub(crate) fn into_slab(self) -> &'a Slab<'a> {
        // SAFETY: The slab contains valid slab data.
        unsafe { self.0.cast().as_ref() }
    }

    pub(crate) fn into_shard(self) -> &'a Shard<'a> {
        &self.into_slab().shards[Slab::HEADER_COUNT]
    }

    pub(crate) fn into_raw(self) -> NonNull<[u8]> {
        NonNull::from_raw_parts(self.0, self.size)
    }

    pub(crate) fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// # Safety
    ///
    /// `ptr` must point to an owned & valid slab.
    pub(crate) unsafe fn from_ptr(ptr: NonNull<()>) -> Self {
        SlabRef(ptr, PhantomData)
    }
}

/// The slab data header, usually consumes a shard.
///
/// # Invariant
///
/// The header resides at the front of the slab memory, which is aligned to
/// [`SLAB_SIZE`], which means every immutable reference or pointer can obtain
/// its root slab reference using pointer masking without violating the
/// ownership rules.
// #[repr(align(4194304))] // SLAB_SIZE
pub(crate) struct Slab<'a> {
    pub(super) thread_id: u64,
    pub(super) arena_id: usize,
    is_huge: bool,
    size: usize,
    used: Cell<usize>,
    abandoned: Cell<usize>,
    pub(super) abandoned_next: AtomicPtr<()>,
    shards: [Shard<'a>; SHARD_COUNT],
}

impl PartialEq for Slab<'_> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl<'a> Slab<'a> {
    pub(crate) const HEADER_COUNT: usize = mem::size_of::<Slab>().div_ceil(SHARD_SIZE);

    pub(crate) fn is_abandoned(&self) -> bool {
        self.abandoned.get() == self.used.get() && self.abandoned.get() > 0
    }

    fn shards(&self) -> impl Iterator<Item = &Shard<'a>> {
        let inner = self.shards[Self::HEADER_COUNT..].iter();
        inner.take_while(|shard| shard.shard_count.get() > 0)
    }

    pub(crate) fn collect_abandoned(&self) -> bool {
        debug_assert!(self.is_abandoned());
        self.shards().any(|shard| {
            let was_unused = shard.is_unused();
            shard.collect(false);
            if !was_unused && shard.is_unused() {
                self.abandoned.set(self.abandoned.get() - 1);
                self.used.set(self.used.get() - 1);
            }
            self.used.get() == 0
        })
    }

    /// # Safety
    ///
    /// The pointer must serve as an immutable reference to a direct/indirect
    /// field in a valid slab reference of `'a`.
    pub(crate) unsafe fn from_ptr<T>(ptr: NonNull<T>) -> NonNull<Self> {
        // SAFETY: `Slab`s are aligned to its size boundary, so we can obtain it
        // directly.
        unsafe { NonNull::new_unchecked(ptr.as_ptr().mask(!(SLAB_SIZE - 1)).cast::<Slab>()) }
    }

    /// The shard area is owned by each corresponding shard in a similar way to
    /// `Cell<[u8; SHARD_SIZE]>`, so the memory can be safely manipulated when
    /// its shard is alive in the scope.
    ///
    /// # Safety
    ///
    /// `this` must points to a valid slab.
    unsafe fn shard_area(this: NonNull<Self>, index: usize) -> NonNull<[u8]> {
        debug_assert!(index < SHARD_COUNT);
        let ptr = this.cast::<u8>();
        // SAFETY: `index` is bounded by the assertion.
        let ptr = unsafe { ptr.add(index * SHARD_SIZE) };
        NonNull::from_raw_parts(ptr.cast(), SHARD_SIZE)
    }

    /// # Safety
    ///
    /// `ptr` must be a pointer within the range of `this`.
    unsafe fn shard_meta(this: NonNull<Self>, ptr: NonNull<()>) -> (NonNull<Shard<'a>>, usize) {
        let shards = ptr::addr_of!((*this.as_ptr()).shards);
        let index = if !unsafe { ptr::addr_of!((*this.as_ptr()).is_huge).read() } {
            (ptr.addr().get() - this.addr().get()) / SHARD_SIZE
        } else {
            Self::HEADER_COUNT
        };
        let shard = shards
            .cast::<Shard<'a>>()
            .with_addr(shards.addr() + index * mem::size_of::<Shard<'a>>());
        (unsafe { NonNull::new_unchecked(shard.cast_mut()) }, index)
    }

    /// # Safety
    ///
    /// `ptr` must be a pointer within the range of `this`.
    pub(crate) unsafe fn shard_infos(
        this: NonNull<Self>,
        ptr: NonNull<()>,
        _layout: Option<Layout>,
    ) -> (NonNull<Shard<'a>>, BlockRef<'a>, usize) {
        // SAFETY: The same as `shard_meta`.
        let (shard, index) = unsafe { Self::shard_meta(this, ptr) };
        debug_assert!(index >= Self::HEADER_COUNT);

        // SAFETY: `this` is valid.
        let area = unsafe { Self::shard_area(this, index) };

        // SAFETY: `AtomicUsize` is `Sync`, so we can load it from any thread.
        // FIXME: Use `atomic_load(*const T, Ordering) -> T` to avoid references.
        let obj_size = unsafe { (*ptr::addr_of!((*shard.as_ptr()).obj_size)).load(Relaxed) };

        let ptr = area.cast::<()>().map_addr(|addr| {
            let offset = ptr.addr().get() - addr.get();
            // SAFETY: `addr` is not zero, and offset is decreased, so the result is not
            // zero.
            unsafe { NonZeroUsize::new_unchecked(addr.get() + (offset - offset % obj_size)) }
        });

        (shard, unsafe { BlockRef::from_raw(ptr) }, obj_size)
    }

    /// # Safety
    ///
    /// `ptr` must be properly aligned to [`SLAB_SIZE`], and owns a freshly
    /// allocated block of memory sized `SLAB_SIZE`.
    pub(crate) unsafe fn init(
        ptr: NonNull<[u8]>,
        thread_id: u64,
        arena_id: usize,
        is_huge: bool,
        free_is_zero: bool,
    ) -> SlabRef<'a> {
        let slab = ptr.cast::<Self>().as_ptr();
        let header_count =
            mem::size_of_val(ptr.cast::<Self>().as_uninit_ref()).div_ceil(SHARD_SIZE);
        assert_eq!(header_count, Self::HEADER_COUNT);

        addr_of_mut!((*slab).thread_id).write(thread_id);
        addr_of_mut!((*slab).arena_id).write(arena_id);
        addr_of_mut!((*slab).is_huge).write(is_huge);
        addr_of_mut!((*slab).size).write(ptr.len());
        addr_of_mut!((*slab).used).write(Cell::new(0));
        addr_of_mut!((*slab).abandoned).write(Cell::new(0));
        addr_of_mut!((*slab).abandoned_next).write(ptr::null_mut::<()>().into());

        let shards: &mut [MaybeUninit<Shard<'a>>; SHARD_COUNT] =
            mem::transmute(addr_of_mut!((*slab).shards).as_uninit_mut().unwrap());

        let (first, next) = shards[Self::HEADER_COUNT..].split_first_mut().unwrap();
        first.write(Shard::new(SHARD_COUNT - Self::HEADER_COUNT, free_is_zero));
        for s in next {
            s.write(Shard::new(0, free_is_zero));
        }

        SlabRef(ptr.cast(), PhantomData)
    }
}

/// A linked list of memory blocks of the same size.
///
/// # Invariant
///
/// This structure cannot be used directly on stack, and must only be referenced
/// from [`Slab`].
#[derive(Default)]
pub(crate) struct Shard<'a> {
    link: CellLink<&'a Self>,
    shard_count: Cell<usize>,

    pub(crate) obj_size: AtomicUsize,
    cap_limit: Cell<usize>,
    capacity: Cell<usize>,

    free: Cell<Option<BlockRef<'a>>>,
    local_free: Cell<Option<BlockRef<'a>>>,
    used: Cell<usize>,
    pub(crate) is_in_full: Cell<bool>,
    free_is_zero: Cell<bool>,

    pub(crate) thread_free: AtomicBlockRef<'a>,
}

impl<'a> PartialEq for &'a Shard<'a> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl<'a> CellLinked for &'a Shard<'a> {
    fn link(&self) -> &CellLink<Self> {
        &self.link
    }
}

impl<'a> Shard<'a> {
    pub(crate) fn new(shard_count: usize, free_is_zero: bool) -> Self {
        Shard {
            shard_count: Cell::new(shard_count),
            free_is_zero: Cell::new(free_is_zero),
            ..Default::default()
        }
    }

    pub(crate) fn is_unused(&self) -> bool {
        self.used.get() == 0
    }

    pub(crate) fn is_full(&self) -> bool {
        self.used.get() == self.capacity.get()
    }

    pub(crate) fn pop_block(&self) -> Option<BlockRef<'a>> {
        self.free.take().map(|mut block| {
            self.used.set(self.used.get() + 1);
            self.free.set(block.next.take());
            block
        })
    }

    pub(crate) fn pop_block_aligned(&self, align: usize) -> Option<BlockRef<'a>> {
        self.free.take().and_then(|mut block| {
            if block.0.is_aligned_to(align) {
                self.used.set(self.used.get() + 1);
                self.free.set(block.next.take());
                Some(block)
            } else {
                self.free.set(Some(block));
                None
            }
        })
    }

    /// # Returns
    ///
    /// `true` if this shard is unused after the deallocation.
    pub(crate) fn push_block(&self, mut block: BlockRef<'a>) -> bool {
        block.next = self.local_free.take();
        self.local_free.set(Some(block));
        let used = self.used.get() - 1;
        self.used.set(used);
        used == 0
    }

    /// This method don't update the usage of the shard, which will be updated
    /// when collecting.
    ///
    /// # Safety
    ///
    /// `this` must point to a valid `Shard<'a>` without any mutable reference.
    pub(crate) unsafe fn push_block_mt(this: NonNull<Self>, mut block: BlockRef<'a>) {
        // SAFETY: `AtomicUsize` is `Sync`, so we can load it from any thread.
        // FIXME: Use `atomic_load(*const T, Ordering) -> T` to avoid references.
        let thread_free = unsafe { &*ptr::addr_of!((*this.as_ptr()).thread_free) };

        let mut cur = thread_free.0.load(Relaxed);
        loop {
            // SAFETY: `thread_free` owns a list of blocks.
            block.next = NonNull::new(cur).map(|p| unsafe { BlockRef::from_raw(p) });

            let new = block.0.as_ptr();
            match thread_free.0.compare_exchange(cur, new, AcqRel, Acquire) {
                Ok(_) => break,
                Err(e) => cur = e,
            }
        }
    }

    fn collect_thread_free(&self) {
        let ptr = self.thread_free.0.swap(ptr::null_mut(), AcqRel);
        let mut thread_free = match NonNull::new(ptr) {
            // SAFETY: `thread_free` owns a list of blocks.
            Some(ptr) => unsafe { BlockRef::from_raw(ptr) },
            _ => return,
        };

        let mut tail = &mut thread_free;
        let mut count = 1;
        *loop {
            match &mut tail.next {
                Some(next) => tail = next,
                slot => break slot,
            }
            count += 1;
        } = self.local_free.take();

        self.local_free.set(Some(thread_free));
        self.used.set(self.used.get() - count);
    }

    pub(crate) fn collect(&self, force: bool) {
        self.collect_thread_free();

        let Some(local_free) = self.local_free.take() else {
            return;
        };
        let free = if let Some(mut free) = self.free.take() {
            if !force {
                self.local_free.set(Some(local_free));
            } else {
                let mut tail = &mut free;
                *loop {
                    match &mut tail.next {
                        Some(next) => tail = next,
                        slot => break slot,
                    }
                } = Some(local_free);
                self.free_is_zero.set(false);
            }
            free
        } else {
            self.free_is_zero.set(false);
            local_free
        };
        self.free.set(Some(free))
    }
}

impl<'a> Shard<'a> {
    pub(crate) fn slab(&self) -> (&'a Slab<'a>, usize) {
        // SAFETY: A shard cannot be manually used on the stack.
        let slab = unsafe { Slab::from_ptr(self.into()) };
        let slab = unsafe { slab.as_ref() };
        // SAFETY: `self` must reside in the `shards` array in its `Slab`.
        let index = unsafe { (self as *const Self).sub_ptr(slab.shards.as_ptr()) };
        (slab, index)
    }

    pub(crate) fn extend_count(&self, count: usize) {
        let (slab, index) = self.slab();
        // SAFETY: `slab` is valid.
        let area = unsafe { Slab::shard_area(slab.into(), index) };
        let obj_size = self.obj_size.load(Relaxed);
        let capacity = self.capacity.get();
        debug_assert!(capacity + count <= self.cap_limit.get());

        // SAFETY: the `area` is owned by this shard in a similar way to `Cell<[u8]>`.
        let iter = (capacity..capacity + count).map(|index| unsafe {
            let ptr = area.cast::<u8>().add(index * obj_size);
            ptr.write_bytes(0, mem::size_of::<Block>());
            BlockRef::from_raw(ptr.cast())
        });

        let mut last = self.free.take();
        iter.rev().for_each(|mut block| {
            block.next = last.take();
            last = Some(block);
        });
        self.free.set(last);

        self.capacity.set(capacity + count);
    }

    pub(crate) fn extend(&self) {
        const MIN_EXTEND: usize = 4;
        const MAX_EXTEND_SIZE: usize = 4096;

        let delta = self.cap_limit.get() - self.capacity.get();
        let limit = (MAX_EXTEND_SIZE / self.obj_size.load(Relaxed)).max(MIN_EXTEND);
        self.extend_count(limit.min(delta))
    }

    pub(crate) fn init_huge(&self, size: usize) {
        debug_assert!(size > SHARD_SIZE);

        let (slab, _) = self.slab();

        slab.used.set(slab.used.get() + 1);
        self.obj_size.store(size, Relaxed);
        self.cap_limit.set(1);
        self.extend_count(1);
    }

    pub(crate) fn init(&self, obj_size: usize) -> Option<&'a Shard<'a>> {
        debug_assert!(obj_size <= SHARD_SIZE);

        let (slab, index) = self.slab();

        let shard_count = self.shard_count.replace(1) - 1;
        let next_shard = (shard_count > 0).then(|| {
            let next_shard = &slab.shards[index + 1];
            next_shard.shard_count.set(shard_count);
            next_shard
        });

        slab.used.set(slab.used.get() + 1);
        let old_obj_size = self.obj_size.swap(obj_size, Relaxed);
        self.cap_limit.set(SHARD_SIZE / obj_size);

        if old_obj_size != obj_size {
            self.free.set(None);
            self.local_free.set(None);
            if old_obj_size != 0 {
                self.free_is_zero.set(false);
            }
            self.extend();
        }

        next_shard
    }

    pub(crate) fn fini(&self) -> Result<Option<&Self>, SlabRef> {
        let (slab, _) = self.slab();
        if self.is_unused() {
            slab.used.set(slab.used.get() - 1);
            self.cap_limit.set(0);
        } else {
            slab.abandoned.set(slab.abandoned.get() + 1);
        }

        if slab.abandoned.get() != slab.used.get() {
            Ok((!slab.abandoned.get() > 0).then_some(self))
        } else {
            Err(SlabRef(NonNull::from(slab).cast(), PhantomData))
        }
    }
}

pub(crate) type ShardList<'a> = CellList<&'a Shard<'a>>;

/// An allocated block before delivered to the user. That is to say, it contains
/// a valid [`Block`].
///
/// The block owns its underlying memory, although the corresponding size is
/// specified by its shard.
#[repr(transparent)]
#[must_use = "blocks must be used"]
pub(crate) struct BlockRef<'a>(NonNull<()>, PhantomData<&'a ()>);

// SAFETY: The block owns its underlying memory.
unsafe impl<'a> Send for BlockRef<'a> {}
unsafe impl<'a> Sync for BlockRef<'a> {}

pub(crate) struct Block<'a> {
    next: Option<BlockRef<'a>>,
}

impl<'a> Deref for BlockRef<'a> {
    type Target = Block<'a>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The block contains a valid block data.
        unsafe { self.0.cast().as_ref() }
    }
}

impl<'a> DerefMut for BlockRef<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The block contains a valid block data.
        unsafe { self.0.cast().as_mut() }
    }
}

impl<'a> BlockRef<'a> {
    #[must_use = "blocks must be used"]
    pub(crate) fn into_raw(self) -> NonNull<()> {
        self.0
    }

    /// # Safety
    ///
    /// The pointer must contain a valid block data.
    pub(crate) unsafe fn from_raw(ptr: NonNull<()>) -> Self {
        BlockRef(ptr, PhantomData)
    }
}

/// An atomic slot containing an `Option<Block<'a>>`.
#[derive(Default)]
#[repr(transparent)]
pub(crate) struct AtomicBlockRef<'a>(AtomicPtr<()>, PhantomData<&'a ()>);

impl<'a> AtomicBlockRef<'a> {}
