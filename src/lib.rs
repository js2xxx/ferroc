#![no_std]
#![feature(allocator_api)]
#![feature(if_let_guard)]
#![feature(isqrt)]
#![feature(let_chains)]
#![feature(non_null_convenience)]
#![feature(pointer_is_aligned)]
#![feature(ptr_as_uninit)]
#![feature(ptr_mask)]
#![feature(ptr_metadata)]
#![feature(ptr_sub_ptr)]
#![feature(slice_ptr_get)]
#![feature(strict_provenance)]

#[cfg(test)]
extern crate alloc;

pub mod arena;
pub mod heap;
mod os;
mod slab;

#[cfg(feature = "os-mmap")]
pub type Global = self::os::mmap::MmapAlloc;
pub use self::os::{Chunk, OsAlloc};

#[cfg(feature = "os-mmap")]
pub type Heap<'a> = self::heap::Heap<'a, Global>;
#[cfg(feature = "os-mmap")]
pub type Context<'a> = self::heap::Context<'a, Global>;
#[cfg(feature = "os-mmap")]
pub type Arenas<'a> = self::arena::Arenas<Global>;
#[cfg(feature = "os-mmap")]
pub type Error<'a> = self::arena::Error<Global>;

#[cfg(test)]
mod test {
    use alloc::vec::Vec;
    use core::{
        alloc::{Allocator, Layout},
        iter,
    };

    use crate::{
        arena::{SHARD_SIZE, SLAB_SIZE},
        Arenas, Context, Heap,
    };

    #[test]
    fn basic() {
        let arena = Arenas::new(Default::default());
        let cx = Context::new(&arena);
        let heap = Heap::new(&cx);

        let mut vec = Vec::new_in(heap.by_ref());
        vec.extend([1, 2, 3, 4]);
        vec.extend([5, 6, 7, 8]);
        assert_eq!(vec.iter().sum::<i32>(), 10 + 26);
        drop(vec);
    }

    #[test]
    fn huge() {
        let arena = Arenas::new(Default::default());
        let cx = Context::new(&arena);
        let heap = Heap::new(&cx);

        let mut vec = Vec::new_in(heap.by_ref());
        vec.extend(iter::repeat(0u8).take(SLAB_SIZE + 5 * SHARD_SIZE));
        vec[SLAB_SIZE / 2] = 123;
        drop(vec)
    }

    #[test]
    fn direct() {
        let arena = Arenas::new(Default::default());
        let cx = Context::new(&arena);
        let heap = Heap::new(&cx);

        let layout = Layout::from_size_align(12345, SLAB_SIZE * 2).unwrap();
        let ptr = heap.allocate(layout).unwrap();
        unsafe { heap.deallocate(ptr.cast(), layout) }
    }
}
