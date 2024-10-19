// #[cfg(feature = "c")]
// use core::num::NonZeroUsize;
use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::{Cell, UnsafeCell},
    marker::PhantomPinned,
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZeroU64,
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicU64, Ordering::*},
};

use super::{Context, Heap};
use crate::{
    arena::Arenas,
    base::{BaseAlloc, Chunk},
};

const POINTER_WIDTH: u32 = usize::BITS;
const BUCKETS: usize = POINTER_WIDTH as usize - 1;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct BucketIndex {
    bucket: usize,
    bucket_count: usize,
    index: usize,
}

impl BucketIndex {
    const fn from_id(id: NonZeroU64) -> Self {
        let bucket = POINTER_WIDTH - id.leading_zeros() - 1;
        let bucket_count = 1 << bucket;
        let index = id.get() - bucket_count;
        debug_assert!(index < isize::MAX as u64);

        BucketIndex {
            bucket: bucket as usize,
            bucket_count: bucket_count as usize,
            index: index as usize,
        }
    }
}

#[repr(align(128))]
struct Entry<'arena, B: BaseAlloc> {
    cx: ManuallyDrop<Context<'arena, B>>,
    heap: ManuallyDrop<Heap<'arena, 'arena, B>>,
    next_reclaimed_id: AtomicU64,
    _marker: PhantomPinned,
}

impl<B: BaseAlloc> Drop for Entry<'_, B> {
    fn drop(&mut self) {
        // SAFETY: The drop order should be explicitly specified: The heap should be
        // dropped before the context.
        unsafe {
            ManuallyDrop::drop(&mut self.heap);
            ManuallyDrop::drop(&mut self.cx);
        }
    }
}

struct Bucket<'arena, B: BaseAlloc> {
    chunk: UnsafeCell<MaybeUninit<Chunk<B>>>,
    pointer: AtomicPtr<Entry<'arena, B>>,
}

impl<'arena, B: BaseAlloc> Bucket<'arena, B> {
    const fn new() -> Self {
        Bucket {
            chunk: UnsafeCell::new(MaybeUninit::uninit()),
            pointer: AtomicPtr::new(ptr::null_mut()),
        }
    }

    #[allow(clippy::declare_interior_mutable_const)]
    const NEW: Self = Bucket::new();

    fn allocate(bi: BucketIndex, arenas: &'arena Arenas<B>) -> Option<Chunk<B>> {
        let layout = Layout::new::<Entry<'arena, B>>();
        let (layout, _) = layout
            .repeat(bi.bucket_count)
            .expect("layout calculation failed: too many bucket requests");
        let chunk = match arenas.base().allocate(layout, true) {
            Ok(chunk) => chunk,
            Err(_err) => {
                #[cfg(feature = "error-log")]
                log::error!("allocation for thread-local failed: too many bucket requests: {_err}");
                return None;
            }
        };
        let mut bucket = chunk.pointer().cast::<Entry<'arena, B>>();
        for _ in 0..bi.bucket_count {
            // SAFETY: All `bucket`s are within the range of the previously allocated chunk,
            // for its layout is calculated before.
            unsafe {
                let td = bucket.as_uninit_mut();
                let td = td.write(Entry {
                    cx: ManuallyDrop::new(Context::new(arenas)),
                    heap: ManuallyDrop::new(Heap::new_uninit()),
                    next_reclaimed_id: AtomicU64::new(0),
                    _marker: PhantomPinned,
                });
                td.heap.init(Pin::new_unchecked(&td.cx));
                bucket = bucket.add(1);
            }
        }
        Some(chunk)
    }

    /// # Safety
    ///
    /// This function is mocking the signature and purpose of `Drop::drop`, but
    /// it needs `bucket_index` for its metadata. Thus, `bucket_index` must
    /// corresponds to the position of the bucket.
    unsafe fn drop(this: &mut Self, bucket_index: usize) -> bool {
        let mut bucket = *this.pointer.get_mut();
        if !bucket.is_null() {
            let bucket_count = 1 << bucket_index;
            for _ in 0..bucket_count {
                unsafe { ptr::drop_in_place(bucket) };
                bucket = unsafe { bucket.add(1) };
            }
            unsafe { this.chunk.get_mut().assume_init_drop() }
            return true;
        }
        false
    }
}

/// A collection for thread-local heaps.
///
/// This structure serves as a substitute for builtin dynamic TLS support with
/// `__tls_get_addr`, which uses its slow `dlmalloc` to allocate TLS variables.
///
/// Most of the direct utilities from this structure is `unsafe`. For usages
/// that are totally safe, see the examples in [`ThreadData`].
pub struct ThreadLocal<'arena, B: BaseAlloc> {
    arena: &'arena Arenas<B>,

    empty_heap: Heap<'arena, 'arena, B>,

    main: UnsafeCell<MaybeUninit<Entry<'arena, B>>>,
    buckets: [Bucket<'arena, B>; BUCKETS],

    next_reclaimed_id: AtomicU64,
    next_id: AtomicU64,

    _marker: PhantomPinned,
}

// SAFETY:
// - For buckets, we only expose thread data entries to its corresponding
//   thread, so there's no data race;
// - For IDs, we use atomics to assign and recycle the thread ids, se there's no
//   data race either;
// - For the empty heap, this structure is immutable since empty heaps cannot
//   allocate anything, and thus cannot deallocate anything either.
unsafe impl<B: BaseAlloc + Sync> Sync for ThreadLocal<'_, B> {}

impl<'arena, B: BaseAlloc> ThreadLocal<'arena, B> {
    /// Creates a collection for thread-local heaps.
    pub const fn new(arena: &'arena Arenas<B>) -> Self {
        Self {
            arena,
            empty_heap: Heap::new_uninit(),
            main: UnsafeCell::new(MaybeUninit::uninit()),
            buckets: [Bucket::NEW; BUCKETS],
            next_reclaimed_id: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
            _marker: PhantomPinned,
        }
    }

    /// The default uninitialized heap, serving as a const initialization
    /// thread-local heap references.
    ///
    /// This method is safe because uninitialized heaps don't mutate their inner
    /// data at all, thus practically `Sync`.
    pub const fn empty_heap<'a>(self: Pin<&'a Self>) -> Pin<&'a Heap<'arena, 'a, B>> {
        // SAFETY: The empty heap is immutable.
        unsafe {
            let pointer = Pin::get_ref(self);
            let new_pointer = &pointer.empty_heap;
            Pin::new_unchecked(new_pointer)
        }
    }
}

impl<'arena, B: BaseAlloc> ThreadLocal<'arena, B> {
    /// Acquire the current thread-local heap with this thread's initialized id.
    ///
    /// # Safety
    ///
    /// `id` must be unique with each live thread regarding only this structure
    /// and may or may not be recycled.
    pub unsafe fn get(self: Pin<&Self>, id: NonZeroU64) -> Pin<&Heap<'arena, '_, B>> {
        let bi = BucketIndex::from_id(id);
        // SAFETY: Any data is not moved.
        unsafe {
            self.map_unchecked(|this| {
                // SAFETY: the thread data entry is initialized in `self.assign`.
                this.get_inner(bi).unwrap_unchecked()
            })
        }
    }

    /// Acquires a new thread id and initialize its associated heap.
    ///
    /// # Invariants
    ///
    /// - The returned id is guaranteed to be unique with each live thread
    ///   regarding only this structure and may or may not be recycled.
    /// - The returned heap is initialized.
    ///
    /// # Examples
    ///
    /// Users can Store the information in its thread-local variable in 2 ways:
    ///
    /// 1. Store the thread ID only, and get access to the heap every time using
    ///    `self.get`;
    /// 2. Store both the thread ID and the pinned heap.
    ///
    /// Both 2 ways needs to run its corresponding destructor manually. If RAII
    /// is preferred, [`ThreadData`] can be used instead.
    ///
    /// # Panics
    ///
    /// Panics if the acquisition failed.
    pub fn assign(self: Pin<&Self>) -> Option<(Pin<&Heap<'arena, '_, B>>, u64)> {
        // SAFETY: `id` is used to initialize its thread data entry below immediately.
        let id = unsafe { self.acquire_id() };
        let heap = match NonZeroU64::new(id) {
            // SAFETY: The id 0 is only assigned once to the main heap, and will never be recycled.
            None => unsafe {
                let pointer = self.main.get();
                let td = (*pointer).write(Entry {
                    cx: ManuallyDrop::new(Context::new(self.arena)),
                    heap: ManuallyDrop::new(Heap::new_uninit()),
                    next_reclaimed_id: AtomicU64::new(0),
                    _marker: PhantomPinned,
                });
                td.heap.init(Pin::new_unchecked(&td.cx));
                Pin::new_unchecked(&*td.heap)
            },
            Some(id) => {
                let bi = BucketIndex::from_id(id);
                // SAFETY: `id` is freshly allocated, which belongs to no other thread.
                //
                // Note that while  freshly allocated, the value of `id` may be reclaimed from
                // another dead thread, which means its thread data entry can be already
                // initialized and should not be `insert`ed unconditionally.
                unsafe {
                    let this = Pin::into_inner_unchecked(self);
                    let heap = this.get_inner(bi).or_else(move || this.insert(bi))?;
                    Pin::new_unchecked(heap)
                }
            }
        };
        Some((heap, id))
    }

    /// # Safety
    ///
    /// The corresponding thread data entry to the returned ID must be
    /// initialized after acquisition.
    unsafe fn acquire_id(self: Pin<&Self>) -> u64 {
        let mut id = self.next_reclaimed_id.load(Relaxed);
        loop {
            let Some(ret) = NonZeroU64::new(id) else {
                const MAX_ID: u64 = i64::MAX as u64;
                const SATURATED_ID: u64 = (MAX_ID & u64::MAX) + ((MAX_ID ^ u64::MAX) >> 1);

                match self.next_id.fetch_add(1, Relaxed) {
                    MAX_ID.. => {
                        self.next_id.store(SATURATED_ID, Relaxed);
                        panic!("Thread ID overflow");
                    }
                    next_id => break next_id,
                }
            };
            let bi = BucketIndex::from_id(ret);
            // SAFETY: Every reclaimed id corresponds to a previously owner thread alongside
            // with its valid thread data entry.
            let td = unsafe { self.entry(bi).unwrap_unchecked() };
            let next = td.next_reclaimed_id.load(Relaxed);

            match self
                .next_reclaimed_id
                .compare_exchange_weak(id, next, AcqRel, Acquire)
            {
                Ok(_) => break ret.get(),
                Err(actual) => id = actual,
            }
        }
    }

    /// Release the current thread-local heap with this thread's id.
    ///
    /// This function may be registered as the thread-local destructor when the
    /// current thread exits.
    ///
    /// # Safety
    ///
    /// - `id` must be previously [assign]ed.
    /// - The current thread must not use this id to access its thread-local
    ///   heap.
    ///
    /// [assign]: ThreadLocal::assign
    pub unsafe fn put(self: Pin<&Self>, id: u64) {
        let Some(id) = NonZeroU64::new(id) else {
            return;
        };
        let mut old = self.next_reclaimed_id.load(Relaxed);
        loop {
            let bi = BucketIndex::from_id(id);
            // SAFETY: The corresponding data is initialized.
            let td = unsafe { self.entry(bi).unwrap_unchecked() };
            td.next_reclaimed_id.store(old, Relaxed);
            match self
                .next_reclaimed_id
                .compare_exchange_weak(old, id.get(), AcqRel, Acquire)
            {
                Ok(_) => return,
                Err(x) => old = x,
            }
        }
    }
}

impl<'arena, B: BaseAlloc> ThreadLocal<'arena, B> {
    #[inline]
    unsafe fn bucket_slot(&self, bi: BucketIndex) -> &Bucket<'arena, B> {
        unsafe { self.buckets.get_unchecked(bi.bucket) }
    }

    #[inline]
    unsafe fn entry(&self, bi: BucketIndex) -> Option<&Entry<'arena, B>> {
        let bucket = unsafe { self.bucket_slot(bi) }.pointer.load(Acquire);
        if bucket.is_null() {
            return None;
        }
        Some(unsafe { &*bucket.add(bi.index) })
    }

    #[inline]
    unsafe fn get_inner(&self, bi: BucketIndex) -> Option<&Heap<'arena, 'arena, B>> {
        Some(&unsafe { self.entry(bi) }?.heap)
    }

    #[cold]
    unsafe fn insert(&self, bi: BucketIndex) -> Option<&Heap<'arena, 'arena, B>> {
        let bucket_slot = unsafe { self.bucket_slot(bi) };
        let bucket = bucket_slot.pointer.load(Acquire);

        let bucket = if bucket.is_null() {
            let chunk = Bucket::allocate(bi, self.arena)?;
            let new_bucket = chunk.pointer().cast();

            match bucket_slot.pointer.compare_exchange(
                ptr::null_mut(),
                new_bucket.as_ptr(),
                AcqRel,
                Acquire,
            ) {
                Ok(_) => {
                    unsafe { (*bucket_slot.chunk.get()).write(chunk) };
                    new_bucket.as_ptr()
                }
                Err(already_init) => already_init,
            }
        } else {
            bucket
        };
        Some(unsafe { &(*bucket.add(bi.index)).heap })
    }
}

impl<B: BaseAlloc> Drop for ThreadLocal<'_, B> {
    fn drop(&mut self) {
        for (index, bucket_slot) in self.buckets.iter_mut().enumerate() {
            // SAFETY: `drop` only during drops. Following the normal drop order.
            if !unsafe { Bucket::drop(bucket_slot, index) } {
                break;
            }
        }
    }
}

/// A thread-local heap allocated from [`ThreadLocal`].
///
/// This structure is practically equivalent to [`Heap`], except being
/// allocated from a dedicated collection of thread-locals to avoid invocation
/// of built-in TLS allocation functions like `__tls_get_addr`.
///
/// # Examples
///
/// Creating thread data on the stack:
///
/// ```
/// #![feature(allocator_api)]
///
/// use core::pin::pin;
/// use ferroc::{
///     arena::Arenas,
///     base::Mmap,
///     heap::{ThreadLocal, ThreadData}
/// };
///
/// let arenas = Arenas::new(Mmap);
/// let thread_local = pin!(ThreadLocal::new(&arenas));
/// let thread_data = ThreadData::new(thread_local.as_ref());
///
/// let mut vec = Vec::with_capacity_in(5, &thread_data);
/// vec.extend([1, 2, 3, 4, 5]);
/// assert_eq!(vec.iter().sum::<i32>(), 15);
/// ```
///
/// Creating on the real thread-local storage:
///
/// ```
/// # #![feature(allocator_api)]
///
/// # use core::pin::Pin;
/// # use ferroc::{
/// #     arena::Arenas,
/// #     base::Mmap,
/// #     heap::{ThreadLocal, ThreadData}
/// # };
///
/// static ARENAS: Arenas<Mmap> = Arenas::new(Mmap);
/// static THREAD_LOCAL: ThreadLocal<Mmap> = ThreadLocal::new(&ARENAS);
///
/// thread_local! {
///     static THREAD_DATA: ThreadData<'static, 'static, Mmap>
///         = ThreadData::new(Pin::static_ref(&THREAD_LOCAL));
/// }
///
/// THREAD_DATA.with(|td| {
///     let mut vec = Vec::with_capacity_in(5, td);
///     vec.extend([1, 2, 3, 4, 5]);
///     assert_eq!(vec.iter().sum::<i32>(), 15);
/// })
/// ```
///
/// While this structure is not `Send` or `Sync`, users can wrap it with a unit
/// struct and forward the (de)allocation functions to `THREAD_DATA.with(|td| /*
/// ... */)`, thus creating a `Send` & `Sync` Allocator.
///
/// **HOWEVER**, Since the [`thread_local!`](std::thread_local) macro in the
/// standard library uses allocation internally, marking the wrapped allocator
/// above as the global allocator will result in infinite recursion. If a global
/// allocator is desired, consider using this crate's [`config`](crate::config)
/// macros instead.
pub struct ThreadData<'t, 'arena, B: BaseAlloc> {
    thread_local: Pin<&'t ThreadLocal<'arena, B>>,
    heap: Cell<Pin<&'t Heap<'arena, 't, B>>>,
    id: Cell<Option<u64>>,
}

impl<'t, 'arena: 't, B: BaseAlloc> ThreadData<'t, 'arena, B> {
    /// Creates a new thread-local heap.
    pub const fn new(thread_local: Pin<&'t ThreadLocal<'arena, B>>) -> Self {
        ThreadData {
            thread_local,
            heap: Cell::new(thread_local.empty_heap()),
            id: Cell::new(None),
        }
    }

    fn init_heap(&self) -> Option<&'t Heap<'arena, 't, B>> {
        let (heap, id) = self.thread_local.assign()?;
        self.heap.set(heap);
        self.id.set(Some(id));
        Some(Pin::get_ref(heap))
    }

    /// Allocate a block of memory from the current thread-local heap.
    ///
    /// See [`Heap::allocate`] for more information.
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<()>, AllocError> {
        self.heap.get().allocate_with(layout, false, unsafe {
            super::AllocateOptions::new(|| self.init_heap())
        })
    }

    /// Allocate a block of memory from the current thread-local heap, and zero
    /// it.
    ///
    /// See [`Heap::allocate_zeroed`] for more information.
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<()>, AllocError> {
        self.heap.get().allocate_with(layout, true, unsafe {
            super::AllocateOptions::new(|| self.init_heap())
        })
    }

    /// Get the layout of a block of memory from the current thread-local heap.
    ///
    /// # Safety
    ///
    /// See [`Heap::layout_of`] for more information.
    pub unsafe fn layout_of(&self, ptr: NonNull<u8>) -> Layout {
        // SAFETY: The pointer is valid .
        unsafe { self.heap.get().layout_of(ptr) }
    }

    /// Deallocate a block of memory to the current thread-local heap.
    ///
    /// # Safety
    ///
    /// See [`Heap::deallocate`] for more information.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The pointer is valid for `layout`.
        unsafe { self.heap.get().deallocate(ptr, layout) }
    }
}

impl<'t, 'arena: 't, B: BaseAlloc> Deref for ThreadData<'t, 'arena, B> {
    type Target = Heap<'arena, 't, B>;

    fn deref(&self) -> &Self::Target {
        Pin::get_ref(self.heap.get())
    }
}

unsafe impl<'t, 'arena: 't, B: BaseAlloc> Allocator for ThreadData<'t, 'arena, B> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
            .map(|ptr| NonNull::slice_from_raw_parts(ptr.cast(), layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.deallocate(ptr, layout) };
    }
}

impl<'t, 'arena: 't, B: BaseAlloc> Drop for ThreadData<'t, 'arena, B> {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            // SAFETY: `id` is previously allocated from `thread_local`, and `heap` is not
            // used any longer.
            unsafe { self.thread_local.put(id) }
        }
    }
}

#[cfg(all(test, feature = "base-mmap"))]
mod tests {
    use core::num::NonZeroU64;

    use crate::heap::thread_local::BucketIndex;

    #[test]
    fn test_bi() {
        assert_eq!(BucketIndex::from_id(NonZeroU64::MIN), BucketIndex {
            bucket: 0,
            bucket_count: 1,
            index: 0,
        });
        assert_eq!(
            BucketIndex::from_id(NonZeroU64::MIN.checked_add(1).unwrap()),
            BucketIndex {
                bucket: 1,
                bucket_count: 2,
                index: 0,
            }
        );
        assert_eq!(
            BucketIndex::from_id(NonZeroU64::MIN.checked_add(2).unwrap()),
            BucketIndex {
                bucket: 1,
                bucket_count: 2,
                index: 1,
            }
        );
        assert_eq!(
            BucketIndex::from_id(NonZeroU64::MIN.checked_add(3).unwrap()),
            BucketIndex {
                bucket: 2,
                bucket_count: 4,
                index: 0,
            }
        );
        assert_eq!(
            BucketIndex::from_id(NonZeroU64::MIN.checked_add(4).unwrap()),
            BucketIndex {
                bucket: 2,
                bucket_count: 4,
                index: 1,
            }
        );
    }

    #[test]
    fn test_thread_data() {
        use core::pin::pin;

        use crate::{
            arena::Arenas,
            base::Mmap,
            heap::{ThreadData, ThreadLocal},
        };

        let arenas = Arenas::new(Mmap);
        let thread_local = pin!(ThreadLocal::new(&arenas));
        let thread_data = ThreadData::new(thread_local.as_ref());

        let mut vec = std::vec::Vec::with_capacity_in(5, &thread_data);
        vec.extend([1, 2, 3, 4, 5]);
        assert_eq!(vec.iter().sum::<i32>(), 15);
    }
}
