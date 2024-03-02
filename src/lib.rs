//! The Ferroc Memory Allocator
//!
//! # Usage
//!
//! Ferroc can be used in many ways by various features. Pick one you prefer:
//!
//! ## The default allocator
//!
//! The simplest use of this crate is to configure it as the default global
//! allocator. Simply add the dependency of this crate, mark an instance of
//! [`Ferroc`] using the `#[global_allocator]` attribute, and done!
//!
//! ```
//! use ferroc::Ferroc;
//!
//! #[global_allocator]
//! static FERROC: Ferroc = Ferroc;
//!
//! let _vec = vec![10; 100];
//! ```
//!
//! Or, if you don't want to use it as the global allocator, [`Ferroc`] provides
//! methods of all the functions you need, as well as the implementation of
//! [`core::alloc::Allocator`]:
//!
//! ```
//! #![feature(allocator_api)]
//! use ferroc::Ferroc;
//!
//! let mut vec = Vec::with_capacity_in(10, Ferroc);
//! vec.extend([1, 2, 3, 4, 5]);
//!
//! let layout = std::alloc::Layout::new::<[u32; 10]>();
//! let memory = Ferroc.allocate(layout).unwrap();
//! unsafe { Ferroc.deallocate(memory.cast(), layout) };
//! ```
//!
//! ## The default allocator (custom configuration)
//!
//! An instance of Ferroc type consists of thread-local contexts and heaps and a
//! global arena collection based on a specified base allocator, which can be
//! configured as you like.
//!
//! To do so, disable the default feature and enable the `global` feature as
//! well as other necessary features to [configure](config) it! Take the
//! embedded use case for example:
//!
//! ```toml
//! [dependencies.ferroc]
//! version = "*"
//! default-features = false
//! features = ["global", "base-static"]
//! ```
//!
//! ```rust,ignore
//! // This is the capacity of the necessary additional static
//! // memory space used by ferroc as the metadata storage.
//! const HEADER_CAP: usize = 4096;
//! ferroc::config!(pub Custom => ferroc::base::Static::<HEADER_CAP>);
//!
//! #[global_allocator]
//! static CUSTOM: Custom = Custom;
//!
//! // Multiple manageable static memory chunks can be loaded at runtime.
//! let chunk = unsafe { Chunk::from_static(/* ... */) };
//! CUSTOM.manage(chunk);
//!
//! // ...And you can start allocation.
//! let _vec = vec![10; 100];
//! ```
//!
//! ## MORE customizations
//!
//! If the configurations of the default allocator can't satisfy your need, you
//! can use the intermediate structures manually while disabling unnecessary
//! features:
//!
//! ```rust
//! #![feature(allocator_api)]
//! use ferroc::{
//!     arena::Arenas,
//!     heap::{Heap, Context},
//!     base::Mmap,
//! };
//!
//! let arenas = Arenas::new(Mmap); // `Arenas` are `Send` & `Sync`...
//! let cx = Context::new(&arenas);
//! let heap = Heap::new(&cx); // ...while `Context`s and `Heap`s are not.
//!
//! // Using the allocator API.
//! let mut vec = Vec::new_in(&heap);
//! vec.extend([1, 2, 3, 4]);
//! assert_eq!(vec.iter().sum::<i32>(), 10);
//!
//! // Manually allocate memory.
//! let layout = std::alloc::Layout::new::<u8>();
//! let ptr = heap.allocate(layout).unwrap();
//! unsafe { heap.deallocate(ptr.cast(), layout) }.unwrap();
//!
//! // Immediately run some delayed clean-up operations.
//! heap.collect(/* force */false);
//! ```
//!
//! ## Using as a dynamic library for `malloc` function series
//!
//! Simply enable the `c` feature and compile it, and you can retrieve the
//! library binary alongside with a `ferroc.h` C/C++ compatible header.
//!
//! If you want to replace the default `malloc` implementation, add a `rustc`
//! flag `--cfg sys_alloc` when compiling.
//!
//! ## Statistics
//!
//! You can get statistics by enabling the `stat` feature and call the `stat`
//! method on the default allocator or other instances like
//! [`Heap`](crate::heap::Heap).
#![no_std]
#![feature(alloc_layout_extra)]
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
#![cfg_attr(feature = "global", allow(internal_features))]
#![cfg_attr(feature = "global", feature(allow_internal_unsafe))]
#![cfg_attr(feature = "global", feature(allow_internal_unstable))]

#[cfg(any(test, miri, feature = "base-mmap"))]
extern crate std;

pub mod arena;
pub mod base;
#[cfg(feature = "c")]
mod c;
#[cfg(feature = "global")]
#[doc(hidden)]
pub mod global;
pub mod heap;
mod slab;
#[cfg(feature = "stat")]
mod stat;
#[doc(hidden)]
pub mod track;

#[cfg(not(feature = "stat"))]
type Stat = ();

#[cfg(feature = "default")]
config_mod!(global_mmap: pub crate::base::Mmap: pthread);

#[cfg(feature = "default")]
pub use self::global_mmap::*;
#[cfg(feature = "stat")]
pub use self::stat::Stat;

#[cfg(test)]
mod test {
    use core::alloc::Layout;
    use std::{thread, vec::Vec};

    #[cfg(not(miri))]
    use crate::arena::SHARD_SIZE;
    use crate::{
        arena::{slab_layout, SLAB_SIZE},
        base::BaseAlloc,
        Ferroc,
    };

    #[test]
    fn basic() {
        let mut vec = Vec::new_in(Ferroc);
        vec.extend([1, 2, 3, 4]);
        vec.extend([5, 6, 7, 8]);
        assert_eq!(vec.iter().sum::<i32>(), 10 + 26);
        drop(vec);
    }

    #[cfg(not(miri))]
    #[test]
    fn large() {
        let mut vec = Vec::new_in(Ferroc);
        vec.extend([0u8; 5 * SHARD_SIZE]);
        vec[2 * SHARD_SIZE] = 123;
        drop(vec);
    }

    #[cfg(not(miri))]
    #[test]
    fn huge() {
        let mut vec = Vec::new_in(Ferroc);
        vec.extend([0u8; SLAB_SIZE + 5 * SHARD_SIZE]);
        vec[SLAB_SIZE / 2] = 123;
        drop(vec);
    }

    #[test]
    fn direct() {
        let layout = Layout::from_size_align(12345, SLAB_SIZE * 2).unwrap();
        let ptr = Ferroc.allocate(layout).unwrap();
        unsafe { Ferroc.deallocate(ptr.cast(), layout) }
    }

    #[test]
    fn manage() {
        let chunk = Ferroc.base().allocate(slab_layout(5), false).unwrap();
        Ferroc.manage(chunk).unwrap();
    }

    #[test]
    fn multithread() {
        let j = thread::spawn(move || {
            let mut vec = Vec::new_in(Ferroc);
            vec.extend([0u8; 100]);
            vec.extend([1; 100]);
            vec
        });
        let vec = j.join().unwrap();
        drop(vec);
    }
}
