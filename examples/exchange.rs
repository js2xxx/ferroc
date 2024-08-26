//! Ported from `xmalloc-test`.
#![feature(allocator_api)]

use std::{
    alloc::Allocator,
    iter,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use ferroc::Ferroc;

// const SIZES: &[usize] = &[
//     8,
//     12,
//     16,
//     24,
//     32,
//     48,
//     64,
//     96,
//     128,
//     192,
//     256,
//     (256 * 3) / 2,
//     512,
//     (512 * 3) / 2,
//     1024,
//     (1024 * 3) / 2,
//     2048,
// ];

const LIMIT: usize = 100;
const BATCH_SIZE: usize = 4096;
const THREADS: usize = 12;
const SLEEP: u64 = 5;

#[global_allocator]
static FERROC: Ferroc = Ferroc;

fn main() {
    println!("Ferroc: {:?}", bench_one(&Ferroc));
    println!("System: {:?}", bench_one(&std::alloc::System));
}

fn bench_one<A: Allocator + Send + Sync>(alloc: &A) -> String {
    let session = Session::new(alloc);
    let start = Instant::now();
    let count: usize = thread::scope(|s| {
        let iter = (0..THREADS).map(|_| {
            let r = s.spawn(|| session.deallocator());
            s.spawn(|| session.allocator());
            r
        });

        let counts: Vec<_> = iter.collect();
        thread::sleep(Duration::from_secs(SLEEP));
        session.stop();

        counts.into_iter().map(|c| c.join().unwrap()).sum()
    });
    format!(
        "count = {count:?}, rate = {:?} frees/sec",
        count as f64 / start.elapsed().as_secs_f64()
    )
}

struct Batch<A: Allocator>(Vec<Box<[u8], A>, A>);

struct Session<'a, A: Allocator> {
    alloc: &'a A,
    stop: AtomicBool,

    batch: Mutex<Vec<Batch<&'a A>, &'a A>>,
    empty: Condvar,
    full: Condvar,
}

impl<'a, A: Allocator> Session<'a, A> {
    fn new(alloc: &'a A) -> Self {
        Self {
            alloc,
            stop: AtomicBool::new(false),
            batch: Mutex::new(Vec::new_in(alloc)),
            empty: Condvar::new(),
            full: Condvar::new(),
        }
    }

    fn push_batch(&self, batch: Batch<&'a A>) {
        let mut batches = self.batch.lock().unwrap();
        while batches.len() >= LIMIT && !self.stop.load(Relaxed) {
            batches = self.full.wait(batches).unwrap();
        }
        batches.push(batch);
        self.empty.notify_one();
    }

    fn pop_batch(&self) -> Option<Batch<&'a A>> {
        let mut batches = self.batch.lock().unwrap();
        while batches.is_empty() && !self.stop.load(Relaxed) {
            batches = self.empty.wait(batches).unwrap();
        }
        let batch = batches.pop();
        self.full.notify_one();
        batch
    }

    fn allocator(&self) {
        while !self.stop.load(Relaxed) {
            let iter = (0..BATCH_SIZE).map(|index| {
                let size = 64;
                let mut vec = Vec::with_capacity_in(size, self.alloc);
                vec.extend(iter::repeat(index as u8).take(size.min(128)));
                vec.into_boxed_slice()
            });
            let mut vec = Vec::with_capacity_in(BATCH_SIZE, self.alloc);
            vec.extend(iter);
            self.push_batch(Batch(vec));
        }
    }

    fn deallocator(&self) -> usize {
        let mut count = 0;
        while !self.stop.load(Relaxed) {
            if let Some(Batch(batch)) = self.pop_batch() {
                count += batch.len();
            }
        }
        count
    }

    fn stop(&self) {
        self.stop.store(true, Relaxed);
        self.empty.notify_all();
        self.full.notify_all();
    }
}
