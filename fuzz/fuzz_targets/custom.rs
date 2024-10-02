#![no_main]
#![feature(allocator_api)]
#![feature(let_chains)]
#![feature(ptr_as_uninit)]
#![feature(ptr_metadata)]

use std::{
    alloc::{Allocator, GlobalAlloc, Layout},
    iter,
    ptr::NonNull,
    sync::Mutex,
    thread,
};

use ferroc::base::{BaseAlloc, Static};
use libfuzzer_sys::fuzz_target;

mod common;
pub use self::common::*;

const HEADER_CAP: usize = 4096;

static STATIC: Static<HEADER_CAP> = Static::new();
ferroc::config_mod!(custom: pub Custom(&STATIC) => &'static Static::<HEADER_CAP>);
use custom::{Chunk, Custom};

#[ctor::ctor]
fn load_memory() {
    use core::cell::UnsafeCell;

    #[repr(align(4194304))]
    struct Memory(UnsafeCell<[u64; 2097152]>);
    unsafe impl Sync for Memory {}
    static STATIC_MEM: Memory = Memory(UnsafeCell::new([0; 2097152]));

    let pointer = unsafe { NonNull::new_unchecked(STATIC_MEM.0.get().cast()) };
    let chunk = unsafe { Chunk::from_static(pointer, Layout::new::<Memory>()) };
    Custom.manage(chunk).unwrap();
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
        Action::Collect { force } => Custom.collect(force),
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
            let size = (size % 10 + 1) as usize;
            let layout = Layout::from_size_align(size << 22, 1 << 22).unwrap();

            if let Ok(chunk) = Custom.base().allocate(layout, false) {
                let _ = Custom.manage(chunk);
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
                NonNull::from_raw_parts(Custom.allocate(layout).ok()?, layout.size())
            }
            (AllocIface::Ferroc, true) => {
                let ptr: NonNull<[u8]> =
                    NonNull::from_raw_parts(Custom.allocate_zeroed(layout).ok()?, layout.size());
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
            (AllocIface::Native, false) => Allocator::allocate(&Custom, layout).ok()?,
            (AllocIface::Native, true) => {
                let ptr = Allocator::allocate_zeroed(&Custom, layout).ok()?;
                assert!(unsafe { ptr.as_ref().iter().all(|&b| b == 0) });
                ptr
            }
            (AllocIface::Global, false) => unsafe {
                let ptr = GlobalAlloc::alloc(&Custom, layout);
                if ptr.is_null() {
                    return None;
                }
                NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size())
            },
            (AllocIface::Global, true) => unsafe {
                let ptr = GlobalAlloc::alloc(&Custom, layout);
                if ptr.is_null() {
                    return None;
                }
                let ptr = NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size());
                assert!(ptr.as_ref().iter().all(|&b| b == 0));
                ptr
            },
        };
        unsafe { ptr.as_uninit_slice_mut()[size / 2].write(align.ilog2() as u8) };
        Some(Allocation { ptr, layout, iface })
    }

    fn check_layout(&self) {
        let req_layout = unsafe { Custom.layout_of(self.ptr.cast()) };
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
                AllocIface::Ferroc => Custom.deallocate(self.ptr.cast(), self.layout),
                AllocIface::Native => Allocator::deallocate(&Custom, self.ptr.cast(), self.layout),
                AllocIface::Global => {
                    GlobalAlloc::dealloc(&Custom, self.ptr.as_ptr().cast(), self.layout)
                }
            }
        }
    }
}
