mod block;
mod cell_link;

use core::{
    cell::Cell,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ops::Deref,
    ptr::{self, addr_of_mut, NonNull},
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering::*},
};

pub(crate) use self::block::BlockRef;
use self::{
    block::AtomicBlockRef,
    cell_link::{CellLink, CellLinked, CellList},
};
use crate::{
    arena::{Error, SHARD_COUNT, SHARD_SIZE, SLAB_SIZE},
    base::BaseAlloc,
    Stat,
};

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
    pub(super) is_large_or_huge: bool,
    pub(super) size: usize,
    used: Cell<usize>,
    abandoned: Cell<usize>,
    pub(super) abandoned_next: AtomicPtr<()>,
    shards: [Shard<'a>; SHARD_COUNT],
}

impl<'a> Slab<'a> {
    pub(crate) const HEADER_COUNT: usize = mem::size_of::<Slab>().div_ceil(SHARD_SIZE);

    pub(crate) fn is_abandoned(&self) -> bool {
        self.abandoned.get() == self.used.get() && self.abandoned.get() > 0
    }

    pub(crate) fn inc_used(&self) {
        self.used.set(self.used.get() + 1);
    }

    fn shards(&self) -> impl Iterator<Item = &Shard<'a>> {
        let inner = self.shards[Self::HEADER_COUNT..].iter();
        inner.take_while(|shard| shard.shard_count.get() > 0)
    }

    pub(crate) fn collect_abandoned(&self) -> bool {
        !self.is_abandoned()
            || self.shards().any(|shard| {
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
    pub(crate) unsafe fn from_ptr<T>(ptr: NonNull<T>) -> Option<NonNull<Self>> {
        NonNull::new(ptr.as_ptr().mask(!(SLAB_SIZE - 1)).cast::<Slab>())
    }

    /// The shard area is owned by each corresponding shard in a similar way to
    /// `Cell<[u8; SHARD_SIZE]>`, so the memory can be safely manipulated when
    /// its shard is alive in the scope.
    ///
    /// # Safety
    ///
    /// `this` must points to a valid slab.
    unsafe fn shard_area(this: NonNull<Self>, index: usize) -> NonNull<u8> {
        debug_assert!(index < SHARD_COUNT);
        let ptr = this.cast::<u8>();
        // SAFETY: `index` is bounded by the assertion.
        unsafe { ptr.add(index * SHARD_SIZE) }
    }

    /// # Safety
    ///
    /// `ptr` must be a pointer within the range of `this`.
    unsafe fn shard_meta(this: NonNull<Self>, ptr: NonNull<()>) -> (NonNull<Shard<'a>>, usize) {
        let shards = ptr::addr_of!((*this.as_ptr()).shards);
        let index = if !unsafe { ptr::addr_of!((*this.as_ptr()).is_large_or_huge).read() } {
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
    ) -> (NonNull<Shard<'a>>, BlockRef<'a>) {
        // SAFETY: The same as `shard_meta`.
        let (shard, index) = unsafe { Self::shard_meta(this, ptr) };
        debug_assert!(index >= Self::HEADER_COUNT);

        // SAFETY: `AtomicUsize` is `Sync`, so we can load it from any thread.
        // FIXME: Use `atomic_load(*const T, Ordering) -> T` to avoid references.
        let ptr = if !unsafe { (*ptr::addr_of!((*shard.as_ptr()).has_aligned)).load(Relaxed) } {
            ptr
        } else {
            let obj_size = unsafe { Shard::obj_size_raw(shard) };

            // SAFETY: `this` is valid.
            let area = unsafe { Self::shard_area(this, index) };

            area.cast::<()>().map_addr(|addr| {
                let offset = ptr.addr().get() - addr.get();
                // SAFETY: `addr` is not zero, and offset is decreased, so the result is not
                // zero.
                unsafe { NonZeroUsize::new_unchecked(addr.get() + (offset - offset % obj_size)) }
            })
        };

        (shard, unsafe { BlockRef::from_raw(ptr) })
    }

    /// # Safety
    ///
    /// `ptr` must be properly aligned to [`SLAB_SIZE`], and owns a freshly
    /// allocated block of memory sized `SLAB_SIZE`.
    pub(crate) unsafe fn init(
        ptr: NonNull<[u8]>,
        thread_id: u64,
        arena_id: usize,
        is_large_or_huge: bool,
        free_is_zero: bool,
    ) -> SlabRef<'a> {
        let slab = ptr.cast::<Self>().as_ptr();
        let header_count =
            mem::size_of_val(ptr.cast::<Self>().as_uninit_ref()).div_ceil(SHARD_SIZE);
        assert_eq!(header_count, Self::HEADER_COUNT);

        addr_of_mut!((*slab).thread_id).write(thread_id);
        addr_of_mut!((*slab).arena_id).write(arena_id);
        addr_of_mut!((*slab).is_large_or_huge).write(is_large_or_huge);
        addr_of_mut!((*slab).size).write(ptr.len());
        addr_of_mut!((*slab).used).write(Cell::new(0));
        addr_of_mut!((*slab).abandoned).write(Cell::new(0));
        addr_of_mut!((*slab).abandoned_next).write(ptr::null_mut::<()>().into());

        let shards: &mut [MaybeUninit<Shard<'a>>; SHARD_COUNT] = mem::transmute(
            addr_of_mut!((*slab).shards)
                .as_uninit_mut()
                .unwrap_unchecked(),
        );

        let (first, rest) = shards[Self::HEADER_COUNT..]
            .split_first_mut()
            .unwrap_unchecked();
        first.write(Shard::new(
            ptr.cast(),
            Self::HEADER_COUNT,
            SHARD_COUNT - Self::HEADER_COUNT,
            free_is_zero,
        ));
        for (index, s) in rest.iter_mut().enumerate() {
            s.write(Shard::new(
                ptr.cast(),
                index + Self::HEADER_COUNT + 1,
                0,
                free_is_zero,
            ));
        }

        SlabRef(ptr.cast(), PhantomData)
    }
}

/// Cache some results of pointer arithmetics for both performance and
/// additional provenance information.
struct ShardHeader<'a> {
    #[cfg(miri)]
    slab: NonNull<Slab<'a>>,
    shard_area: NonNull<u8>,
    marker: PhantomData<&'a ()>,
}

impl<'a> Default for ShardHeader<'a> {
    fn default() -> Self {
        Self {
            #[cfg(miri)]
            slab: NonNull::dangling(),
            shard_area: NonNull::dangling(),
            marker: PhantomData,
        }
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
    header: ShardHeader<'a>,

    link: CellLink<'a, Self>,
    shard_count: Cell<usize>,
    is_committed: Cell<bool>,

    pub(crate) obj_size: AtomicUsize,
    cap_limit: Cell<usize>,
    capacity: Cell<usize>,

    free: Cell<Option<BlockRef<'a>>>,
    local_free: Cell<Option<BlockRef<'a>>>,
    used: Cell<usize>,
    pub(crate) is_in_full: Cell<bool>,
    pub(crate) has_aligned: AtomicBool,
    free_is_zero: Cell<bool>,

    thread_free: AtomicBlockRef<'a>,
}

impl<'a> CellLinked<'a> for Shard<'a> {
    fn link(&'a self) -> &'a CellLink<'a, Self> {
        &self.link
    }
}

impl<'a> Shard<'a> {
    pub(crate) unsafe fn new(
        slab: NonNull<Slab<'a>>,
        index: usize,
        shard_count: usize,
        free_is_zero: bool,
    ) -> Self {
        Shard {
            header: ShardHeader {
                #[cfg(miri)]
                slab,
                shard_area: Slab::shard_area(slab, index),
                marker: PhantomData,
            },
            shard_count: Cell::new(shard_count),
            free_is_zero: Cell::new(free_is_zero),
            ..Default::default()
        }
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.shard_count.get()
    }

    pub(crate) fn is_unused(&self) -> bool {
        self.used.get() == 0
    }

    pub(crate) fn is_full(&self) -> bool {
        self.used.get() == self.capacity.get()
    }

    pub(crate) unsafe fn obj_size_raw(this: NonNull<Self>) -> usize {
        unsafe { (*ptr::addr_of!((*this.as_ptr()).obj_size)).load(Relaxed) }
    }

    pub(crate) fn pop_block(&self) -> Option<(BlockRef<'a>, bool)> {
        self.free.take().map(|mut block| {
            self.used.set(self.used.get() + 1);
            self.free.set(block.take_next());
            (block, self.free_is_zero.get())
        })
    }

    pub(crate) fn pop_block_aligned(&self, align: usize) -> Option<(BlockRef<'a>, bool)> {
        self.free.take().and_then(|mut block| {
            if block.as_ptr().is_aligned_to(align) {
                self.used.set(self.used.get() + 1);
                self.free.set(block.take_next());
                Some((block, self.free_is_zero.get()))
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
        block.set_next(self.local_free.take());
        self.local_free.set(Some(block));
        self.used.set(self.used.get() - 1);
        self.used.get() == 0
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
        let thread_free = unsafe { &*ptr::addr_of!((*this.as_ptr()).thread_free) }.get();

        let mut cur = thread_free.load(Relaxed);
        loop {
            // SAFETY: `thread_free` owns a list of blocks.
            block.set_next(NonNull::new(cur).map(|p| unsafe { BlockRef::from_raw(p) }));

            let new = block.as_ptr();
            match thread_free.compare_exchange_weak(cur, new, AcqRel, Acquire) {
                Ok(_) => break,
                Err(e) => cur = e,
            }
        }
    }

    fn collect_thread_free(&self) {
        let mut ptr = self.thread_free.get().load(Relaxed);
        let mut thread_free = loop {
            match NonNull::new(ptr) {
                None => return,
                Some(nn) => match self.thread_free.get().compare_exchange_weak(
                    ptr,
                    ptr::null_mut(),
                    AcqRel,
                    Acquire,
                ) {
                    Ok(_) => break unsafe { BlockRef::from_raw(nn) },
                    Err(e) => ptr = e,
                },
            }
        };

        let count = thread_free.set_tail(self.local_free.take());
        self.local_free.set(Some(thread_free));
        self.used.set(self.used.get() - count);
    }

    pub(crate) fn collect(&self, force: bool) -> bool {
        self.collect_thread_free();

        let Some(local_free) = self.local_free.take() else {
            return !self.is_full();
        };
        let free = if let Some(mut free) = self.free.take() {
            if !force {
                self.local_free.set(Some(local_free));
            } else {
                free.set_tail(Some(local_free));
                self.free_is_zero.set(false);
            }
            free
        } else {
            self.free_is_zero.set(false);
            local_free
        };
        self.free.set(Some(free));
        true
    }
}

impl<'a> Shard<'a> {
    pub(crate) fn slab(&self) -> (&'a Slab<'a>, usize) {
        // SAFETY: A shard cannot be manually used on the stack.
        //
        // Every shard only associates with one slab, which resides exactly on the
        // address by pointer arithmetics. Miri doesn't recognize this, so we provide it
        // with additional provenance information.
        let slab = unsafe { Slab::from_ptr(self.into()).unwrap_unchecked() };
        #[cfg(not(miri))]
        let slab = unsafe { slab.as_ref() };
        #[cfg(miri)]
        let slab = unsafe {
            assert_eq!(slab, self.header.slab);
            self.header.slab.as_ref()
        };
        // SAFETY: `self` must reside in the `shards` array in its `Slab`.
        let index = unsafe { (self as *const Self).sub_ptr(slab.shards.as_ptr()) };
        (slab, index)
    }

    fn extend_inner(&self, obj_size: usize, capacity: usize, cap_limit: usize) {
        if cap_limit == capacity {
            return;
        }
        const MIN_EXTEND: usize = 4;
        const MAX_EXTEND_SIZE: usize = 4096;

        let limit = (MAX_EXTEND_SIZE / obj_size).max(MIN_EXTEND);

        let count = limit.min(cap_limit - capacity);
        debug_assert!(count > 0);
        debug_assert!(capacity + count <= cap_limit);

        let area = self.header.shard_area;
        // SAFETY: the `area` is owned by this shard in a similar way to `Cell<[u8]>`.
        let iter = (capacity..capacity + count)
            .map(|index| unsafe { BlockRef::new(area.add(index * obj_size).cast()) });

        let mut last = self.free.take();
        iter.rev().for_each(|mut block| {
            block.set_next(last.take());
            last = Some(block);
        });
        self.free.set(last);

        self.capacity.set(capacity + count);
    }

    #[inline]
    pub(crate) fn extend(&self) {
        self.extend_inner(
            self.obj_size.load(Relaxed),
            self.capacity.get(),
            self.cap_limit.get(),
        )
    }

    pub(crate) fn init_large_or_huge<B: BaseAlloc>(
        &self,
        obj_size: usize,
        slab_count: NonZeroUsize,
        base: &B,
        _stat: &mut Stat,
    ) -> Result<(), Error<B>> {
        debug_assert!(obj_size > SHARD_SIZE);

        #[cfg(feature = "stat")]
        {
            _stat.shards += 1;
        }

        self.obj_size.store(obj_size, Relaxed);
        let usable_size = SLAB_SIZE * slab_count.get() - SHARD_SIZE * Slab::HEADER_COUNT;
        let cap_limit = usable_size / obj_size;
        self.cap_limit.set(cap_limit);
        self.has_aligned.store(false, Relaxed);

        self.capacity.set(0);
        self.free.set(None);
        self.local_free.set(None);
        self.used.set(0);

        if !self.is_committed.replace(true) {
            let area = self.header.shard_area;
            unsafe { base.commit(NonNull::from_raw_parts(area.cast(), usable_size)) }
                .map_err(Error::Commit)?;
        }

        self.extend_inner(obj_size, 0, cap_limit);

        Ok(())
    }

    pub(crate) fn init<B: BaseAlloc>(
        &self,
        obj_size: usize,
        base: &B,
        _stat: &mut Stat,
    ) -> Result<Option<&'a Shard<'a>>, Error<B>> {
        debug_assert!(obj_size <= SLAB_SIZE / 2);

        if obj_size > SHARD_SIZE {
            self.init_large_or_huge(obj_size, NonZeroUsize::MIN, base, _stat)?;
            return Ok(None);
        }

        #[cfg(feature = "stat")]
        {
            _stat.shards += 1;
        }

        if !self.is_committed.replace(true) {
            let area = self.header.shard_area;
            unsafe { base.commit(NonNull::from_raw_parts(area.cast(), SHARD_SIZE)) }
                .map_err(Error::Commit)?;
        }

        let old_obj_size = self.obj_size.swap(obj_size, Relaxed);
        let cap_limit = SHARD_SIZE / obj_size;
        self.cap_limit.set(cap_limit);
        self.has_aligned.store(false, Relaxed);

        let (slab, index) = self.slab();
        let shard_count = self.shard_count.replace(1) - 1;
        debug_assert!(shard_count + index < SHARD_COUNT);
        let next_shard = (shard_count > 0)
            .then(|| &slab.shards[index + 1])
            .inspect(|next_shard| next_shard.shard_count.set(shard_count));

        if old_obj_size != obj_size {
            self.capacity.set(0);
            self.free.set(None);
            self.local_free.set(None);
            self.used.set(0);
            if old_obj_size != 0 {
                self.free_is_zero.set(false);
            }
            self.extend_inner(obj_size, 0, cap_limit);
        }

        Ok(next_shard)
    }

    pub(crate) fn fini(&self, _stat: &mut Stat) -> Result<Option<&Self>, SlabRef> {
        debug_assert!(!self.link.is_linked());

        let (slab, _) = self.slab();
        if self.is_unused() {
            slab.used.set(slab.used.get() - 1);
            self.cap_limit.set(0);
            #[cfg(feature = "stat")]
            {
                _stat.shards -= 1;
            }
        } else {
            slab.abandoned.set(slab.abandoned.get() + 1);
            #[cfg(feature = "stat")]
            {
                _stat.abandoned_shards += 1;
            }
        }

        if slab.abandoned.get() != slab.used.get() {
            Ok((!slab.abandoned.get() > 0).then_some(self))
        } else {
            Err(SlabRef(NonNull::from(slab).cast(), PhantomData))
        }
    }
}

pub(crate) type ShardList<'a> = CellList<'a, Shard<'a>>;
