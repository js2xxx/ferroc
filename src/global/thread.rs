#[cfg(feature = "libc")]
#[allow(unused_imports)]
pub use libc::{pthread_key_create, pthread_key_delete, pthread_setspecific};

#[macro_export]
#[doc(hidden)]
#[allow_internal_unsafe]
#[allow_internal_unstable(thread_local)]
macro_rules! thread_statics {
    () => {
        use core::{cell::Cell, num::NonZeroU64, pin::Pin, ptr};

        use super::{Error, Heap, THREAD_LOCALS};

        #[thread_local]
        static HEAP: Cell<Pin<&Heap>> = Cell::new(THREAD_LOCALS.uninit_heap());

        pub fn with<T: core::fmt::Debug>(f: impl FnOnce(&Heap) -> T) -> T {
            f(&HEAP.get())
        }

        pub fn with_lazy<T: core::fmt::Debug>(
            mut f: impl FnMut(&Heap) -> Result<T, Error>,
        ) -> Result<T, Error> {
            match f(&HEAP.get()) {
                Ok(t) => Ok(t),
                Err(Error::Uninit) => {
                    let (heap, id) = Pin::static_ref(&THREAD_LOCALS).assign();
                    unsafe { init(id) };
                    HEAP.set(heap);
                    f(&heap)
                }
                Err(err) => Err(err),
            }
        }
    };
}

#[cfg(feature = "libc")]
#[macro_export]
#[doc(hidden)]
macro_rules! thread_init_pthread {
    () => {
        unsafe fn init(id: NonZeroU64) {
            unsafe extern "C" fn fini(id: *mut core::ffi::c_void) {
                if let Some(id) = NonZeroU64::new(id.addr().try_into().unwrap()) {
                    HEAP.set(THREAD_LOCALS.uninit_heap());
                    Pin::static_ref(&THREAD_LOCALS).put(id);
                }
            }

            let data = ptr::without_provenance_mut(id.get().try_into().unwrap());
            register_thread_dtor(data, fini);
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
                let ret = $crate::global::thread::pthread_key_create(&mut new, Some(dtor));
                assert_eq!(ret, 0);
                key = new as usize;

                if let Err(already_set) = KEY_INIT.compare_exchange(VAL_INIT, key, AcqRel, Acquire)
                {
                    let _ = $crate::global::thread::pthread_key_delete(key as _);
                    key = already_set;
                }
            }
            let ret = $crate::global::thread::pthread_setspecific(key as _, data);
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
        unsafe fn init(_id: NonZeroU64) {}
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
