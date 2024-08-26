//! Ported from `mstress`.
#![feature(allocator_api)]

use std::{
    alloc::Allocator,
    iter, mem,
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

use ferroc::Ferroc;

const COOKIE: usize = 0xbf58476d1ce4e5b9;
const THREADS: usize = 6;
#[cfg(not(any(miri, debug_assertions)))]
const SCALE: usize = 5000;
#[cfg(any(miri, debug_assertions))]
const SCALE: usize = 10;
#[cfg(not(any(miri, debug_assertions)))]
const ITER: usize = 25;
#[cfg(any(miri, debug_assertions))]
const ITER: usize = 1;
#[cfg(not(any(miri, debug_assertions)))]
const TRANSFER_COUNT: usize = 1000;
#[cfg(any(miri, debug_assertions))]
const TRANSFER_COUNT: usize = 20;

#[cfg(feature = "track-valgrind")]
#[global_allocator]
static FERROC: Ferroc = Ferroc;

fn main() {
    println!("ferroc: {:?}", do_bench(&Ferroc));
    #[cfg(not(any(feature = "track-valgrind", any(miri, debug_assertions))))]
    println!("system: {:?}", do_bench(&std::alloc::System));
}

fn do_bench<A: Allocator + Send + Sync>(a: &A) -> Duration {
    let mut transfer: Vec<_> = iter::repeat_with(|| Mutex::new(None))
        .take(TRANSFER_COUNT)
        .collect();
    let start = Instant::now();
    for _i in 0..ITER {
        thread::scope(|s| {
            let transfer = &transfer;
            for tid in 0..THREADS {
                s.spawn(move || bench_one(tid, transfer, a));
            }
        });
        (transfer.iter_mut().filter(|_| probably(50))).for_each(|t| *t.get_mut().unwrap() = None)
    }
    drop(transfer);
    start.elapsed()
}

fn bench_one<'a, A: Allocator>(tid: usize, transfer: &[Mutex<Option<Items<&'a A>>>], a: &'a A) {
    let mut alloc_count: usize = SCALE * (tid % 8 + 1);
    let mut retain_count: usize = alloc_count / 2;

    let mut retained = Vec::with_capacity_in(retain_count, a);
    let mut data = Vec::new_in(a);

    while alloc_count > 0 || retain_count > 0 {
        if retain_count == 0 || (probably(50) && alloc_count > 0) {
            data.push(Some(Items::new(1 << fastrand::u32(0..5), a)));
            alloc_count -= 1;
        } else {
            retained.push(Some(Items::new(1 << fastrand::u32(0..5), a)));
            retain_count -= 1;
        }

        if probably(67) && !data.is_empty() {
            let index = fastrand::usize(0..data.len());
            data[index] = None;
        }

        if probably(25) && !data.is_empty() {
            let di = fastrand::usize(0..data.len());
            let ti = fastrand::usize(0..transfer.len());
            mem::swap(&mut data[di], &mut transfer[ti].lock().unwrap());
        }
    }
}

#[inline]
fn probably(p: u8) -> bool {
    fastrand::u8(0..=100) <= p
}

struct Items<A: Allocator>(Box<[usize], A>);

impl<A: Allocator> Items<A> {
    fn new(count: usize, a: A) -> Self {
        let count = if probably(1) {
            if probably(1) {
                count * 10000
            } else if probably(10) {
                count * 1000
            } else {
                count * 100
            }
        } else {
            count
        };

        let mut vec = Vec::new_in(a);
        vec.extend((0..count).map(|i| (count - i) ^ COOKIE));
        Items(vec.into_boxed_slice())
    }
}

impl<A: Allocator> Drop for Items<A> {
    fn drop(&mut self) {
        for (index, &value) in self.0.iter().enumerate() {
            assert_eq!(
                value ^ COOKIE,
                (self.0.len() - index),
                "memory corruption at block {:p} at {index}",
                self.0
            )
        }
    }
}
