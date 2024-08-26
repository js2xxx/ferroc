// #[cfg(feature = "c")]
// use core::num::NonZeroUsize;
use core::{
    alloc::Layout,
    cell::UnsafeCell,
    marker::PhantomPinned,
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZeroU64,
    pin::Pin,
    ptr,
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

    fn id_in_bucket(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.bucket_count).map(move |i| (self.bucket_count + i) as u64)
    }
}

#[repr(align(128))]
struct ThreadData<'arena, B: BaseAlloc> {
    cx: ManuallyDrop<Context<'arena, B>>,
    heap: ManuallyDrop<Heap<'arena, 'arena, B>>,
    next_reclaimed_id: AtomicU64,
    _marker: PhantomPinned,
}

impl<'arena, B: BaseAlloc> Drop for ThreadData<'arena, B> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.heap);
            ManuallyDrop::drop(&mut self.cx);
        }
    }
}

struct Bucket<'arena, B: BaseAlloc> {
    chunk: UnsafeCell<MaybeUninit<Chunk<B>>>,
    pointer: AtomicPtr<ThreadData<'arena, B>>,
}

impl<'arena, B: BaseAlloc> Bucket<'arena, B> {
    const fn new() -> Self {
        Bucket {
            chunk: UnsafeCell::new(MaybeUninit::uninit()),
            pointer: AtomicPtr::new(ptr::null_mut()),
        }
    }

    const NEW: Self = Bucket::new();

    fn allocate(bi: BucketIndex, arenas: &'arena Arenas<B>) -> Chunk<B> {
        let layout = Layout::new::<ThreadData<'arena, B>>();
        let (layout, _) = layout
            .repeat(bi.bucket_count)
            .expect("layout calculation failed: too many bucket requests");
        let Ok(chunk) = arenas.base().allocate(layout, true) else {
            unreachable!("allocation for thread-local failed: too many bucket requests")
        };
        let mut bucket = chunk.pointer().cast::<ThreadData<'arena, B>>();
        for id in bi.id_in_bucket() {
            unsafe {
                let td = bucket.as_uninit_mut();
                let td = td.write(ThreadData {
                    cx: ManuallyDrop::new(Context::new_with_id(arenas, id)),
                    heap: ManuallyDrop::new(Heap::new_uninit()),
                    next_reclaimed_id: AtomicU64::new(0),
                    _marker: PhantomPinned,
                });
                td.heap.init(&td.cx);
                bucket = bucket.add(1);
            }
        }
        chunk
    }

    unsafe fn drop(this: &mut Self, bucket_index: usize) -> bool {
        let mut bucket = *this.pointer.get_mut();
        if !bucket.is_null() {
            let bucket_count = 1 << bucket_index;
            for _ in 0..bucket_count {
                unsafe { ptr::drop_in_place(bucket) };
                bucket = bucket.add(1);
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
pub struct ThreadLocal<'arena, B: BaseAlloc> {
    arena: &'arena Arenas<B>,

    empty_heap: Heap<'arena, 'arena, B>,
    buckets: [Bucket<'arena, B>; BUCKETS],

    next_reclaimed_id: AtomicU64,
    next_id: AtomicU64,

    _marker: PhantomPinned,
}

unsafe impl<'arena, B: BaseAlloc + Sync> Sync for ThreadLocal<'arena, B> {}

impl<'arena, B: BaseAlloc> ThreadLocal<'arena, B> {
    /// Creates a collection for thread-local heaps.
    pub const fn new(arena: &'arena Arenas<B>) -> Self {
        Self {
            arena,
            empty_heap: Heap::new_uninit(),
            buckets: [Bucket::NEW; BUCKETS],
            next_reclaimed_id: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
            _marker: PhantomPinned,
        }
    }

    /// The default uninitialized heap, serving as a const initialization
    /// thread-local heap references.
    ///
    /// This method is safe because uninitialized heap doesn't mutate its inner
    /// data at all, thus ideomatically `Sync`.
    pub const fn empty_heap(&'static self) -> Pin<&'static Heap<'static, 'static, B>> {
        Pin::static_ref(&self.empty_heap)
    }
}

impl<'arena, B: BaseAlloc> ThreadLocal<'arena, B> {
    /// Acquire the current thread-local heap with this thread's initialized id.
    ///
    /// # Safety
    ///
    /// `id` must be unique with each live thread regarding only this structure
    /// and may or may not be recycled.
    #[inline(always)]
    pub unsafe fn get(self: Pin<&Self>, id: NonZeroU64) -> Pin<&Heap<'arena, '_, B>> {
        let bi = BucketIndex::from_id(id);
        self.map_unchecked(|this| {
            if let Some(heap) = this.get_inner(bi) {
                return heap;
            }
            this.insert(bi)
        })
    }

    /// Acquires a new thread id and initialize its associated heap.
    ///
    /// # Panics
    ///
    /// Panics if the acquisition failed.
    pub fn assign(self: Pin<&Self>) -> (Pin<&Heap<'arena, '_, B>>, NonZeroU64) {
        let id = unsafe { self.acquire_id() };
        (unsafe { self.get(id) }, id)
    }

    unsafe fn acquire_id(self: Pin<&Self>) -> NonZeroU64 {
        let mut id = self.next_reclaimed_id.load(Relaxed);
        loop {
            let Some(ret) = NonZeroU64::new(id) else {
                const MAX_ID: u64 = i64::MAX as u64;
                const SATURATED_ID: u64 = (MAX_ID & u64::MAX) + ((MAX_ID ^ u64::MAX) >> 1);

                break match self.next_id.fetch_add(1, Relaxed) {
                    SATURATED_ID.. => {
                        self.next_id.store(SATURATED_ID, Relaxed);
                        panic!("Thread ID overflow");
                    }
                    next_id => unsafe { NonZeroU64::new_unchecked(next_id) },
                };
            };
            let bi = BucketIndex::from_id(ret);
            let td = self.thread_data(bi).unwrap_unchecked();
            let next = td.next_reclaimed_id.load(Relaxed);

            match self
                .next_reclaimed_id
                .compare_exchange_weak(id, next, AcqRel, Acquire)
            {
                Ok(_) => break ret,
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
    /// - `id` must corresponds to a initialized thread data which the current
    ///   thread is using.
    /// - The current thread must not use this id to access its thread-local
    ///   heap.
    pub unsafe fn put(self: Pin<&Self>, id: NonZeroU64) {
        let mut old = self.next_reclaimed_id.load(Relaxed);
        loop {
            let bi = BucketIndex::from_id(id);
            let td = self.thread_data(bi).unwrap_unchecked();
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
        self.buckets.get_unchecked(bi.bucket)
    }

    #[inline]
    unsafe fn thread_data(&self, bi: BucketIndex) -> Option<&ThreadData<'arena, B>> {
        let bucket = self.bucket_slot(bi).pointer.load(Acquire);
        if bucket.is_null() {
            return None;
        }
        Some(&*bucket.add(bi.index))
    }

    #[inline]
    unsafe fn get_inner(&self, bi: BucketIndex) -> Option<&Heap<'arena, 'arena, B>> {
        Some(&self.thread_data(bi)?.heap)
    }

    #[cold]
    unsafe fn insert(&self, bi: BucketIndex) -> &Heap<'arena, 'arena, B> {
        let bucket_slot = self.bucket_slot(bi);
        let bucket = bucket_slot.pointer.load(Acquire);

        let bucket = if bucket.is_null() {
            let chunk = Bucket::allocate(bi, self.arena);
            let new_bucket = chunk.pointer().cast();

            match bucket_slot.pointer.compare_exchange(
                ptr::null_mut(),
                new_bucket.as_ptr(),
                AcqRel,
                Acquire,
            ) {
                Ok(_) => {
                    (*bucket_slot.chunk.get()).write(chunk);
                    new_bucket.as_ptr()
                }
                Err(already_init) => already_init,
            }
        } else {
            bucket
        };
        &(*bucket.add(bi.index)).heap
    }
}

impl<'arena, B: BaseAlloc> Drop for ThreadLocal<'arena, B> {
    fn drop(&mut self) {
        for (index, bucket_slot) in self.buckets.iter_mut().enumerate() {
            if !unsafe { Bucket::drop(bucket_slot, index) } {
                break;
            }
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
}
