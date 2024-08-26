#![no_main]
#![feature(let_chains)]
#![feature(ptr_as_uninit)]

use std::{alloc::Layout, iter, ptr::NonNull, sync::Mutex, thread};

use ferroc::Ferroc;
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};

const THREADS: usize = 12;
const TRANSFER_COUNT: usize = 1000;

#[derive(Debug, Arbitrary)]
enum Action {
    Allocate { size: u32, align_shift: u8 },
    Deallocate { index: u8 },
    LayoutOf { index: u8 },
    Collect { force: bool },
    Transfer { from: u8, to: u8 },
}

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
        Action::Allocate { size, align_shift } => {
            let align_shift = align_shift % 24;
            let size = size % 16777216 + 1;
            let align = 1 << align_shift;
            // eprintln!("actual size = {size:#x}, align = {align:#x}");

            if let Some(a) = Allocation::new(size as usize, align) {
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
    });
}

struct Allocation {
    ptr: NonNull<[u8]>,
    layout: Layout,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(size: usize, align: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, align).unwrap();
        if let Ok(ptr) = Ferroc.allocate(layout) {
            unsafe { ptr.as_uninit_slice_mut()[size / 2].write(align.ilog2() as u8) };
            return Some(Allocation { ptr, layout });
        }
        None
    }

    fn check_layout(&self) {
        let req_layout = unsafe { Ferroc.layout_of(self.ptr.cast()) }.unwrap();
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
        unsafe { Ferroc.deallocate(self.ptr.cast(), self.layout) };
    }
}
