#![no_main]
#![feature(allocator_api)]
#![feature(let_chains)]
#![feature(ptr_as_uninit)]
#![feature(ptr_metadata)]

use std::{
    alloc::{Allocator, Layout},
    iter,
    pin::pin,
    ptr::NonNull,
    sync::{atomic::Ordering::Relaxed, Mutex},
    thread,
};

use ferroc::{
    arena::Arenas,
    base::{BaseAlloc, Mmap},
    heap::{ThreadData, ThreadLocal},
};
use libfuzzer_sys::fuzz_target;

mod common;
pub use self::common::*;

scoped_tls::scoped_thread_local!(static THREAD_DATA: ThreadData<'static, 'static, Mmap>);

fuzz_target!(|action_sets: [Vec<Action>; THREADS]| {
    let base = Mmap::new();
    let arenas = Arenas::new(base);
    let thread_local = pin!(ThreadLocal::new(&arenas));

    let main_td = ThreadData::new(thread_local.as_ref());
    let main_td = unsafe {
        core::mem::transmute::<
            ferroc::heap::ThreadData<'_, '_, ferroc::base::Mmap>,
            ferroc::heap::ThreadData<'_, '_, ferroc::base::Mmap>,
        >(main_td)
    };

    THREAD_DATA.set(&main_td, || {
        let transfers: Vec<_> = iter::repeat_with(|| Mutex::new(None))
            .take(TRANSFER_COUNT)
            .collect();

        thread::scope(|s| {
            for actions in action_sets {
                s.spawn(|| {
                    let td = ThreadData::new(thread_local.as_ref());
                    let td = unsafe {
                        core::mem::transmute::<
                            ferroc::heap::ThreadData<'_, '_, ferroc::base::Mmap>,
                            ferroc::heap::ThreadData<'_, '_, ferroc::base::Mmap>,
                        >(td)
                    };
                    THREAD_DATA.set(&td, || fuzz_one(&arenas, actions, &transfers))
                });
            }
        });
    })
});

fn fuzz_one(arenas: &Arenas<Mmap>, actions: Vec<Action>, transfers: &[Mutex<Option<Allocation>>]) {
    let mut allocations = Vec::new();

    actions.into_iter().for_each(|action| match action {
        Action::Allocate { size, align_shift, zeroed, iface } => {
            let align_shift = align_shift % 19;
            let size = size % 16777216 + 1;
            let align = 1 << align_shift;
            // eprintln!("actual size = {size:#x}, align = {align:#x}");

            if let Some(a) = Allocation::new(size as usize, align, zeroed, iface) {
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

                let chunk = arenas.base().allocate(layout, false).unwrap();
                arenas.manage(chunk).unwrap();
            }
        }
    });
}

struct Allocation {
    ptr: NonNull<[u8]>,
    layout: Layout,
    iface: AllocIface,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(size: usize, align: usize, zeroed: bool, iface: AllocIface) -> Option<Self> {
        let layout = Layout::from_size_align(size, align).unwrap();
        let ptr = match (iface, zeroed) {
            (AllocIface::Ferroc, false) => NonNull::from_raw_parts(
                THREAD_DATA.with(|td| td.allocate(layout)).ok()?,
                layout.size(),
            ),
            (AllocIface::Ferroc, true) => {
                let ptr: NonNull<[u8]> = NonNull::from_raw_parts(
                    THREAD_DATA.with(|td| td.allocate_zeroed(layout)).ok()?,
                    layout.size(),
                );
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
            (AllocIface::Native | AllocIface::Global, false) => THREAD_DATA
                .with(|td| Allocator::allocate(td, layout))
                .ok()?,
            (AllocIface::Native | AllocIface::Global, true) => {
                let ptr = THREAD_DATA
                    .with(|td| Allocator::allocate_zeroed(td, layout))
                    .ok()?;
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
        };
        unsafe { ptr.as_uninit_slice_mut()[size / 2].write(align.ilog2() as u8) };
        Some(Allocation { ptr, layout, iface })
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

        match self.iface {
            AllocIface::Ferroc => unsafe {
                THREAD_DATA.with(|td| td.deallocate(self.ptr.cast(), self.layout))
            },
            AllocIface::Native | AllocIface::Global => unsafe {
                THREAD_DATA.with(|td| Allocator::deallocate(td, self.ptr.cast(), self.layout))
            },
        }
    }
}
