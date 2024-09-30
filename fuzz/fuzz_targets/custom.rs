#![no_main]
#![feature(allocator_api)]
#![feature(let_chains)]
#![feature(ptr_as_uninit)]
#![feature(ptr_metadata)]

use std::{
    alloc::{Allocator, Global, Layout},
    iter,
    pin::Pin,
    ptr::NonNull,
    sync::{atomic::Ordering::Relaxed, Mutex},
    thread,
};

use ferroc::{
    arena::Arenas,
    base::{Chunk, Static},
    heap::{ThreadData, ThreadLocal},
};
use libfuzzer_sys::fuzz_target;

mod common;
pub use self::common::*;

const HEADER_CAP: usize = 4096;
static STATIC: Static<HEADER_CAP> = Static::new();

static ARENAS: Arenas<&'static Static<HEADER_CAP>> = Arenas::new(&STATIC);
static THREAD_LOCAL: ThreadLocal<&'static Static<HEADER_CAP>> = ThreadLocal::new(&ARENAS);

thread_local! {
    static THREAD_DATA: ThreadData<'static, 'static, &'static Static<HEADER_CAP>>
        = ThreadData::new(Pin::static_ref(&THREAD_LOCAL));
}

fuzz_target!(|action_sets: [Vec<Action>; THREADS]| {
    let transfers: Vec<_> = iter::repeat_with(|| Mutex::new(None))
        .take(TRANSFER_COUNT)
        .collect();

    thread::scope(|s| {
        for actions in action_sets {
            s.spawn(|| fuzz_one(actions, &transfers));
        }
    });
});

fn fuzz_one(actions: Vec<Action>, transfers: &[Mutex<Option<Allocation>>]) {
    let mut allocations = Vec::new();

    actions.into_iter().for_each(|action| match action {
        Action::Allocate { size, align_shift, zeroed, .. } => {
            let align_shift = align_shift % 19;
            let size = size % 16777216 + 1;
            let align = 1 << align_shift;
            // eprintln!("actual size = {size:#x}, align = {align:#x}");

            if let Some(a) = Allocation::new(size as usize, align, zeroed) {
                allocations.push(a);
            }
        }
        Action::Deallocate { index } => {
            if let Some(index) = (index as usize).checked_rem(allocations.len()) {
                drop(allocations.swap_remove(index));
            }
        }
        Action::LayoutOf { index } => {
            if let Some(index) = (index as usize).checked_rem(allocations.len()) {
                allocations[index].check_layout();
            }
        }
        Action::Collect { force } => THREAD_DATA.with(|td| td.collect(force)),
        Action::Transfer { from, to } => {
            if let Some(from) = (from as usize).checked_rem(allocations.len())
                && let Some(to) = (to as usize).checked_rem(transfers.len())
            {
                let a = allocations.swap_remove(from);
                let o = transfers[to].lock().unwrap().replace(a);
                if let Some(a) = o {
                    allocations.push(a);
                }
            }
        }
        Action::Manage { size } => {
            let managed = MANAGED.load(Relaxed);
            if managed < MAX_MANAGED
                && MANAGED
                    .compare_exchange(managed, managed + 1, Relaxed, Relaxed)
                    .is_ok()
            {
                let size = (size % 10 + 1) as usize;
                let layout = Layout::from_size_align(size << 22, 1 << 22).unwrap();

                let ptr = Global.allocate(layout).unwrap();
                let chunk = unsafe { Chunk::from_static(ptr.cast(), layout) };
                ARENAS.manage(chunk).unwrap();
            }
        }
    });
}

struct Allocation {
    ptr: NonNull<[u8]>,
    layout: Layout,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(size: usize, align: usize, zeroed: bool) -> Option<Self> {
        let layout = Layout::from_size_align(size, align).unwrap();
        let ptr = match zeroed {
            false => NonNull::from_raw_parts(
                THREAD_DATA.with(|td| td.allocate(layout)).ok()?,
                layout.size(),
            ),
            true => {
                let ptr: NonNull<[u8]> = NonNull::from_raw_parts(
                    THREAD_DATA.with(|td| td.allocate_zeroed(layout)).ok()?,
                    layout.size(),
                );
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
        };
        unsafe { ptr.as_uninit_slice_mut()[size / 2].write(align.ilog2() as u8) };
        Some(Allocation { ptr, layout })
    }

    fn check_layout(&self) {
        let req_layout = unsafe { THREAD_DATA.with(|td| td.layout_of(self.ptr.cast())) };
        assert!(
            req_layout.size() >= self.layout.size(),
            "ptr = {:p}\nreq = {:#x?}\nl = {:#x?}",
            self.ptr,
            req_layout,
            self.layout
        );
        assert!(
            req_layout.align() >= self.layout.align(),
            "ptr = {:p}\nreq = {:#x?}\nl = {:#x?}",
            self.ptr,
            req_layout,
            self.layout
        );
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        assert_eq!(
            unsafe { self.ptr.as_uninit_slice()[self.layout.size() / 2].assume_init_read() },
            self.layout.align().ilog2() as u8,
        );

        unsafe { THREAD_DATA.with(|td| td.deallocate(self.ptr.cast(), self.layout)) }
    }
}
