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
#![feature(strict_provenance)]
#![cfg_attr(all(feature = "global", feature = "base-mmap"), feature(thread_local))]

#[cfg(test)]
extern crate std;

pub mod arena;
pub mod base;
#[cfg(feature = "c")]
mod c;
#[cfg(feature = "global")]
#[cfg_attr(feature = "base-mmap", path = "global-mmap.rs")]
mod global;
pub mod heap;
mod slab;

#[cfg(feature = "global")]
#[allow(unused_imports)]
pub use self::global::*;

#[cfg(test)]
mod test {
    use core::alloc::Layout;
    use std::{thread, vec};

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

    #[test]
    fn multithread() {
        let mut vec = vec![0u8; 100];
        vec.extend([1; 100]);
        let j = thread::spawn(move || drop(vec));
        j.join().unwrap();
    }
}
