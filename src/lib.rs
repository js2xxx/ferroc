#![cfg_attr(not(feature = "os-mmap"), no_std)]
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
#![feature(strict_provenance)]

#[cfg(test)]
extern crate alloc;

pub mod arena;
#[cfg(feature = "os-mmap")]
mod global;
pub mod heap;
mod os;
mod slab;

#[cfg(feature = "os-mmap")]
pub use self::global::*;
pub use self::os::{Chunk, OsAlloc};

#[cfg(test)]
mod test {
    use alloc::vec;
    use core::alloc::{Allocator, Layout};

    use crate::{
        arena::{SHARD_SIZE, SLAB_SIZE},
        Ferroc,
    };

    #[global_allocator]
    static FERROC: Ferroc = Ferroc;

    #[test]
    fn basic() {
        let mut vec = vec![1, 2, 3, 4];
        vec.extend([5, 6, 7, 8]);
        assert_eq!(vec.iter().sum::<i32>(), 10 + 26);
        drop(vec);
    }

    #[test]
    fn huge() {
        let mut vec = vec![0u8; SLAB_SIZE + 5 * SHARD_SIZE];
        vec[SLAB_SIZE / 2] = 123;
        drop(vec)
    }

    #[test]
    fn direct() {
        let layout = Layout::from_size_align(12345, SLAB_SIZE * 2).unwrap();
        let ptr = FERROC.allocate(layout).unwrap();
        unsafe { FERROC.deallocate(ptr.cast(), layout) }
    }
}
