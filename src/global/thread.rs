#[cfg(feature = "libc")]
#[allow(unused_imports)]
pub use libc::{pthread_key_create, pthread_key_delete, pthread_setspecific};

#[macro_export]
#[doc(hidden)]
#[allow_internal_unsafe]
#[allow_internal_unstable(thread_local)]
macro_rules! thread_statics {
    () => {
        use core::{cell::Cell, mem::MaybeUninit};

        use super::{Context, Heap, ARENAS};

        #[thread_local]
        static INIT: Cell<bool> = Cell::new(false);

        #[thread_local]
        static mut CX: MaybeUninit<Context> = MaybeUninit::uninit();

        #[thread_local]
        static mut HEAP: MaybeUninit<Heap> = MaybeUninit::uninit();

        #[inline]
        pub fn with<T>(f: impl FnOnce(&Heap) -> T) -> T {
            if !INIT.get() {
                unsafe {
                    init();
                    let cx = CX.write(Context::new(&ARENAS));
                    HEAP.write(Heap::new(cx));
                }
                INIT.set(true);
            }
            f(unsafe { HEAP.assume_init_ref() })
        }
    };
}

#[cfg(feature = "libc")]
#[macro_export]
#[doc(hidden)]
macro_rules! thread_init_pthread {
    () => {
        unsafe fn init() {
            unsafe extern "C" fn fini(_: *mut core::ffi::c_void) {
                if INIT.get() {
                    HEAP.assume_init_drop();
                    CX.assume_init_drop();
                }
            }

            register_thread_dtor(core::ptr::NonNull::dangling().as_ptr(), fini);
        }

        unsafe fn register_thread_dtor(
            data: *mut core::ffi::c_void,
            dtor: unsafe extern "C" fn(*mut core::ffi::c_void),
        ) {
            use core::sync::atomic::{AtomicU32, Ordering::*};

            static KEY_INIT: AtomicU32 = AtomicU32::new(u32::MAX);
            let mut key = KEY_INIT.load(Relaxed);
            if key == u32::MAX {
                let ret = $crate::global::thread::pthread_key_create(&mut key, Some(dtor));
                assert_eq!(ret, 0);

                if let Err(already_set) = KEY_INIT.compare_exchange(u32::MAX, key, AcqRel, Acquire)
                {
                    let _ = $crate::global::thread::pthread_key_delete(key);
                    key = already_set;
                }
            }
            let ret = $crate::global::thread::pthread_setspecific(key, data);
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
        unsafe fn init() {}
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
