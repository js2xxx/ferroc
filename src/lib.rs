#![cfg_attr(not(feature = "std"), no_std)]
#![feature(allocator_api)]
#![feature(let_chains)]
#![feature(non_null_convenience)]
#![feature(pointer_is_aligned)]
#![feature(ptr_as_uninit)]
#![feature(ptr_mask)]
#![feature(ptr_metadata)]
#![feature(ptr_sub_ptr)]
#![feature(slice_ptr_get)]
#![feature(strict_provenance)]

pub mod arena;
pub mod heap;
mod os;
mod slab;

#[cfg(feature = "std")]
pub use os::mmap::MmapAlloc;

#[cfg(test)]
mod test {
    use core::alloc::Allocator;

    use crate::{
        arena::{slab_layout, Arena},
        heap::{Context, Heap},
        os::OsAlloc,
        MmapAlloc,
    };

    #[test]
    fn basic() {
        let chunk = MmapAlloc.allocate(slab_layout(2)).unwrap();
        let arena = Arena::new(chunk);
        let cx = Context::new(&arena);
        let heap = Heap::new(&cx);

        let mut vec = Vec::new_in(heap.by_ref());
        vec.extend([1, 2, 3, 4]);
        vec.extend([5, 6, 7, 8]);
        assert_eq!(vec.iter().sum::<i32>(), 10 + 26);
        drop(vec);
    }
}
