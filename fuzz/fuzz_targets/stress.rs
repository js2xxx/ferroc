#![no_main]
#![feature(allocator_api)]
#![feature(let_chains)]
#![feature(ptr_as_uninit)]
#![feature(ptr_metadata)]

use std::{
    alloc::{Allocator, Layout},
    iter,
    ptr::NonNull,
    sync::{atomic::Ordering::Relaxed, Mutex},
    thread,
};

use ferroc::{base::BaseAlloc, Ferroc};
use libfuzzer_sys::fuzz_target;

mod common;
pub use self::common::*;

#[global_allocator]
static FERROC: Ferroc = Ferroc;

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
        Action::Collect { force } => Ferroc.collect(force),
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
                let chunk = Ferroc.base().allocate(layout, false).unwrap();
                Ferroc.manage(chunk).unwrap();
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
            (AllocIface::Ferroc, false) => {
                NonNull::from_raw_parts(Ferroc.allocate(layout).ok()?, layout.size())
            }
            (AllocIface::Ferroc, true) => {
                let ptr: NonNull<[u8]> =
                    NonNull::from_raw_parts(Ferroc.allocate_zeroed(layout).ok()?, layout.size());
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
            (AllocIface::Native, false) => <Ferroc as Allocator>::allocate(&Ferroc, layout).ok()?,
            (AllocIface::Native, true) => {
                let ptr = <Ferroc as Allocator>::allocate_zeroed(&Ferroc, layout).ok()?;
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
            (AllocIface::Global, false) => std::alloc::Global.allocate(layout).ok()?,
            (AllocIface::Global, true) => {
                let ptr = std::alloc::Global.allocate_zeroed(layout).ok()?;
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
        };
        unsafe { ptr.as_uninit_slice_mut()[size / 2].write(align.ilog2() as u8) };
        Some(Allocation { ptr, layout, iface })
    }

    fn check_layout(&self) {
        let req_layout = unsafe { Ferroc.layout_of(self.ptr.cast()) };
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

        unsafe {
            match self.iface {
                AllocIface::Ferroc => Ferroc.deallocate(self.ptr.cast(), self.layout),
                AllocIface::Native => {
                    <Ferroc as Allocator>::deallocate(&Ferroc, self.ptr.cast(), self.layout)
                }
                AllocIface::Global => std::alloc::Global.deallocate(self.ptr.cast(), self.layout),
            }
        }
    }
}
