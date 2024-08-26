#![no_main]
#![feature(ptr_as_uninit)]

use std::alloc::Layout;

use ferroc::{
    arena::Arenas,
    heap::{Context, Heap},
    MmapAlloc,
};
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};

#[derive(Debug, Arbitrary)]
enum Action {
    Allocate { size: u32, align_shift: u8 },
    Deallocate { index: u8 },
}

fuzz_target!(|actions: Vec<Action>| {
    let arena = Arenas::new(MmapAlloc);
    let cx = Context::new(&arena);
    let heap = Heap::new(&cx);

    let mut allocations = Vec::new();

    for action in actions {
        match action {
            Action::Allocate { size, align_shift } => {
                let size = size % 131072 + 1;
                let align = 1 << (align_shift % 16);
                // eprintln!("actual size = {size:#x}, align = {align:#x}");

                let layout = Layout::from_size_align(size as usize, align).unwrap();
                if let Ok(ptr) = heap.allocate(layout) {
                    unsafe { ptr.as_uninit_slice_mut()[size as usize / 2].write(align_shift % 16) };
                    allocations.push((ptr, layout));
                }
            }
            Action::Deallocate { index } => {
                if (index as usize) < allocations.len() {
                    let (ptr, layout) = allocations.swap_remove(index as usize);
                    assert_eq!(
                        unsafe { ptr.as_uninit_slice()[layout.size() / 2].assume_init_read() },
                        layout.align().ilog2() as u8,
                    );
                    unsafe { heap.deallocate(ptr.cast(), layout) };
                }
            }
        }
    }

    for (ptr, layout) in allocations {
        assert_eq!(
            unsafe { ptr.as_uninit_slice()[layout.size() / 2].assume_init_read() },
            layout.align().ilog2() as u8,
        );
        unsafe { heap.deallocate(ptr.cast(), layout) };
    }
});
