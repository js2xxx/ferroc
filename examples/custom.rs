use std::{alloc::Layout, cell::UnsafeCell, ptr::NonNull};

use ferroc::base::Static;

// This is the capacity of the necessary additional static
// memory space used by ferroc as the metadata storage.
const HEADER_CAP: usize = 4096;
static STATIC: Static<HEADER_CAP> = Static::new();

ferroc::config!(pub Custom(&STATIC) => &'static Static::<HEADER_CAP>);

#[global_allocator]
static CUSTOM: Custom = Custom;

#[ctor::ctor]
fn load_memory() {
    #[repr(align(4194304))]
    struct Memory(UnsafeCell<[usize; 1024]>);
    unsafe impl Sync for Memory {}
    static STATIC_MEM: Memory = Memory(UnsafeCell::new([0; 1024]));

    let pointer = unsafe { NonNull::new_unchecked(STATIC_MEM.0.get().cast()) };
    let chunk = unsafe { Chunk::from_static(pointer, Layout::new::<Memory>()) };
    Custom.manage(chunk).unwrap();
}

fn main() {
    let mut vec = vec![];
    vec.extend([1, 2, 3, 4, 5]);
    assert_eq!(vec.iter().sum::<i32>(), 15);
    println!("{vec:?}");
}
