#[cfg(feature = "libc")]
#[allow(unused_imports)]
pub use libc::{pthread_key_create, pthread_key_delete, pthread_setspecific};

#[macro_export]
#[doc(hidden)]
#[allow_internal_unsafe]
#[allow_internal_unstable(thread_local)]
#[allow_internal_unstable(const_pin)]
macro_rules! thread_statics {
    () => {
        use core::{cell::Cell, pin::Pin, ptr};

        use super::{Heap, THREAD_LOCALS};

        #[thread_local]
        static HEAP: Cell<Pin<&Heap<'_, '_>>> =
            Cell::new(Pin::static_ref(&THREAD_LOCALS).empty_heap());

        #[inline(always)]
        pub fn with<T>(f: impl FnOnce(&Heap<'_, '_>) -> T) -> T {
            f(&HEAP.get())
        }

        #[inline(always)]
        pub fn with_lazy<T, F>(f: F) -> T
        where
            F: for<'a, 'h> FnOnce(&'a Heap<'h, 'h>, fn() -> Option<&'a Heap<'h, 'h>>) -> T,
        {
            fn fallback<'a>() -> Option<&'a Heap<'static, 'static>> {
                let (heap, id) = Pin::static_ref(&THREAD_LOCALS).assign()?;
                // SAFETY: `init` is called only once for every thread-local heap.
                unsafe { init(id) };
                HEAP.set(heap);
                Some(Pin::get_ref(heap))
            }

            f(Pin::get_ref(HEAP.get()), fallback)
        }
    };
}

#[cfg(feature = "libc")]
#[macro_export]
#[doc(hidden)]
macro_rules! thread_init_pthread {
    () => {
        /// # Safety
        ///
        /// This function must be called only once for every initialized thread-local
        /// heap.
        unsafe fn init(id: u64) {
            /// # Safety
            ///
            /// - `id` must be a valid thread-local heap ID;
            /// - the thread-local heap must not be used after calling this function.
            unsafe extern "C" fn fini(id: *mut core::ffi::c_void) {
                HEAP.set(core::pin::Pin::static_ref(&THREAD_LOCALS).empty_heap());
                let id = id.addr().try_into().unwrap();
                // SAFETY: `id` is valid; the thread-local heap is no longer used.
                unsafe { Pin::static_ref(&THREAD_LOCALS).put(id) };
            }

            let data = ptr::without_provenance_mut(id.try_into().unwrap());
            unsafe { register_thread_dtor(data, fini) };
        }

        unsafe fn register_thread_dtor(
            data: *mut core::ffi::c_void,
            dtor: unsafe extern "C" fn(*mut core::ffi::c_void),
        ) {
            use core::sync::atomic::{AtomicUsize, Ordering::*};

            const VAL_INIT: usize = usize::MAX;
            static KEY_INIT: AtomicUsize = AtomicUsize::new(VAL_INIT);

            let mut key = KEY_INIT.load(Relaxed);
            if key == VAL_INIT {
                let mut new = 0;
                // SAFETY: `dtor` is valid.
                let ret =
                    unsafe { $crate::global::thread::pthread_key_create(&mut new, Some(dtor)) };
                assert_eq!(ret, 0);
                key = new as usize;

                if let Err(already_set) = KEY_INIT.compare_exchange(VAL_INIT, key, AcqRel, Acquire)
                {
                    // SAFETY: `key` is no longer used and reassigned.
                    let _ = unsafe { $crate::global::thread::pthread_key_delete(key as _) };
                    key = already_set;
                }
            }
            // SAFETY: `key` and `data` is valid.
            let ret = unsafe { $crate::global::thread::pthread_setspecific(key as _, data) };
            assert_eq!(ret, 0);
        }
    };
}

#[cfg(not(feature = "libc"))]
#[macro_export]
#[doc(hidden)]
macro_rules! thread_init_pthread {
    () => {
        $crate::thread_init!();
        compile_error!(
            "cannot use `pthread` thread-local destructors while not enabling feature `libc`"
        );
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! thread_init {
    () => {
        unsafe fn init(_id: u64) {}
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! thread_mod {
    () => {
        mod thread {
            $crate::thread_statics!();
            $crate::thread_init!();
        }
    };
    (pthread) => {
        mod thread {
            $crate::thread_statics!();
            $crate::thread_init_pthread!();
        }
    };
}
