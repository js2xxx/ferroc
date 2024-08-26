use core::{
    cell::Cell,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering::*},
};

use super::{Context, Heap, ARENAS};

#[thread_local]
static INIT: Cell<bool> = Cell::new(false);

#[thread_local]
static mut CX: MaybeUninit<Context> = MaybeUninit::uninit();

#[thread_local]
static mut HEAP: MaybeUninit<Heap> = MaybeUninit::uninit();

pub fn with<T>(f: impl FnOnce(&Heap) -> T) -> T {
    let ptr = INIT.as_ptr();
    if !unsafe { ptr.read() } {
        unsafe { init() };
        unsafe { ptr.write(true) };
    }
    f(unsafe { HEAP.assume_init_ref() })
}

unsafe fn init() {
    unsafe extern "C" fn fini(_: *mut libc::c_void) {
        if INIT.get() {
            HEAP.assume_init_drop();
            CX.assume_init_drop();
        }
    }
    debug_assert!(!INIT.get());
    register_thread_dtor(NonNull::dangling().as_ptr(), fini);

    let cx = CX.write(Context::new(&ARENAS));
    HEAP.write(Heap::new(cx));
}

unsafe fn register_thread_dtor(
    data: *mut libc::c_void,
    dtor: unsafe extern "C" fn(*mut libc::c_void),
) {
    static KEY_INIT: AtomicU32 = AtomicU32::new(u32::MAX);
    let mut key = KEY_INIT.load(Relaxed);
    if key == u32::MAX {
        let ret = libc::pthread_key_create(&mut key, Some(dtor));
        assert_eq!(ret, 0);

        if let Err(already_set) = KEY_INIT.compare_exchange(u32::MAX, key, AcqRel, Acquire) {
            let _ = libc::pthread_key_delete(key);
            key = already_set;
        }
    }
    let ret = libc::pthread_setspecific(key, data);
    assert_eq!(ret, 0);
}
