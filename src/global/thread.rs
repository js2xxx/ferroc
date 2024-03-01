#[cfg(feature = "libc")]
#[allow(unused_imports)]
pub use libc::{pthread_key_create, pthread_key_delete, pthread_setspecific};

#[macro_export]
#[doc(hidden)]
#[allow_internal_unsafe]
#[allow_internal_unstable(thread_local)]
macro_rules! thread_statics {
    () => {
        use core::mem::{ManuallyDrop, MaybeUninit};

        use super::{Context, Heap, ARENAS};

        #[thread_local]
        static mut CX: MaybeUninit<Context> = MaybeUninit::uninit();

        #[thread_local]
        static mut HEAP: ManuallyDrop<Heap> = ManuallyDrop::new(Heap::new_uninit());

        #[inline]
        pub fn with_init<T>(f: impl FnOnce(&Heap) -> T) -> T {
            debug_assert!(unsafe { !HEAP.is_init() });
            unsafe {
                init();
                let cx = CX.write(Context::new(&ARENAS));
                HEAP.init(cx);
            }
            f(unsafe { &*core::ptr::addr_of!(HEAP) })
        }

        #[inline]
        pub fn with<T>(f: impl FnOnce(&Heap) -> T) -> Option<T> {
            if unsafe { HEAP.is_init() } {
                return Some(f(unsafe { &*core::ptr::addr_of!(HEAP) }));
            }
            None
        }

        pub fn with_uninit<T>(f: impl FnOnce(&Heap) -> T) -> T {
            f(unsafe { &*core::ptr::addr_of!(HEAP) })
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
                if HEAP.is_init() {
                    ManuallyDrop::drop(&mut *core::ptr::addr_of_mut!(HEAP));
                    CX.assume_init_drop();
                }
            }

            register_thread_dtor(core::ptr::NonNull::dangling().as_ptr(), fini);
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
