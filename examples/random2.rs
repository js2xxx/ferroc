//! Ported from `larson-sized`.
#![feature(allocator_api)]

use std::{
    alloc::Allocator,
    iter::{self, Sum},
    ops::{Add, Range},
    sync::atomic::{AtomicBool, Ordering::*},
    thread::{self, Scope},
    time::{Duration, Instant},
};

use ferroc::Ferroc;

const BLOCK_SIZE: Range<usize> = 8..1000;
const ROUND: usize = 100;
#[cfg(not(miri))]
const BLOCK_COUNT: usize = 5000;
#[cfg(miri)]
const BLOCK_COUNT: usize = 20;
#[cfg(not(miri))]
const THREADS: usize = 12;
#[cfg(miri)]
const THREADS: usize = 2;
const SLEEP: Duration = Duration::from_secs(5);

#[cfg(not(miri))]
#[global_allocator]
static FERROC: Ferroc = Ferroc;

fn main() {
    println!("ferroc: {:.3?}", do_bench(&Ferroc));
    #[cfg(not(any(feature = "track-valgrind", miri)))]
    println!("system: {:.3?}", do_bench(&std::alloc::System));
}

fn do_bench<A: Allocator + Send + Sync>(a: &A) -> SumData {
    let mut array = vec![None; THREADS * BLOCK_COUNT];
    let mut td = [ThreadData::default(); THREADS];
    let stop = AtomicBool::new(false);

    warm_up(&mut array, a);
    let start = Instant::now();
    thread::scope(|s| {
        let stop = &stop;
        for (array, td) in array.chunks_mut(BLOCK_COUNT).zip(&mut td) {
            s.spawn(move || bench_one(s, array, td, stop, a));
        }
        thread::sleep(SLEEP);
        stop.store(true, Relaxed);
    });
    drop(array);
    let dur = start.elapsed();
    let sum: ThreadData = td.into_iter().sum();

    SumData {
        alloc_rate: sum.alloc_count as f64 / dur.as_secs_f64(),
        free_rate: sum.free_count as f64 / dur.as_secs_f64(),
    }
}

fn bench_one<'scope, 'env, 'alloc: 'env, A: Allocator + Send + Sync>(
    s: &'scope Scope<'scope, 'env>,
    array: &'env mut [Option<Vec<u8, &'alloc A>>],
    td: &'env mut ThreadData,
    stop: &'env AtomicBool,
    a: &'alloc A,
) {
    if stop.load(Relaxed) {
        return;
    }
    td.thread_count += 1;

    for _ in 0..BLOCK_COUNT * ROUND {
        let index = fastrand::usize(..array.len());
        if array[index].take().is_some() {
            td.free_count += 1;
        }
        let size = fastrand::usize(BLOCK_SIZE);
        array[index] = Some(new_vec(size, a));
        td.alloc_count += 1;

        if stop.load(Relaxed) {
            return;
        }
    }

    s.spawn(move || bench_one(s, array, td, stop, a));
}

fn warm_up<'a, A: Allocator>(array: &mut [Option<Vec<u8, &'a A>>], a: &'a A) {
    array.iter_mut().for_each(|x| {
        let size = fastrand::usize(BLOCK_SIZE);
        *x = Some(new_vec(size, a))
    });
    fastrand::shuffle(array);
    for _ in 0..BLOCK_COUNT {
        let index = fastrand::usize(..array.len());
        let size = fastrand::usize(BLOCK_SIZE);
        array[index] = Some(new_vec(size, a));
    }
}

fn new_vec<A: Allocator>(size: usize, a: A) -> Vec<u8, A> {
    let mut v = Vec::with_capacity_in(size, a);
    v.extend(iter::repeat_n(0, size));
    v
}

#[derive(Debug, Clone, Copy, Default)]
struct ThreadData {
    alloc_count: usize,
    free_count: usize,
    thread_count: usize,
}

impl Add for ThreadData {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ThreadData {
            alloc_count: self.alloc_count + rhs.alloc_count,
            free_count: self.free_count + rhs.free_count,
            thread_count: self.thread_count + rhs.thread_count,
        }
    }
}

impl Sum for ThreadData {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), Add::add)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SumData {
    alloc_rate: f64,
    free_rate: f64,
}
