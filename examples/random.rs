use std::{
    alloc::{GlobalAlloc, Layout},
    iter,
    marker::PhantomData,
    ptr::NonNull,
    thread,
    time::{Duration, Instant},
};

use ferroc::Ferroc;

#[cfg(not(any(miri, feature = "track-valgrind")))]
const ALL_ROUND: usize = 10000;
#[cfg(any(miri, feature = "track-valgrind"))]
const ALL_ROUND: usize = 10;
const THREAD_COUNT: usize = 6;
const ROUND: usize = ALL_ROUND / THREAD_COUNT;

const BENCH_ARGS: &[BenchArg] = &[
    BenchArg { size: 8, count: 1000 },
    BenchArg { size: 16, count: 5000 },
    BenchArg { size: 48, count: 1000 },
    BenchArg { size: 72, count: 100 },
    BenchArg { size: 148, count: 100 },
    BenchArg { size: 200, count: 100 },
    BenchArg { size: 520, count: 10 },
    BenchArg { size: 1056, count: 5 },
    BenchArg { size: 4096, count: 3 },
    BenchArg { size: 9162, count: 1 },
    BenchArg { size: 34562, count: 1 },
    BenchArg { size: 168524, count: 1 },
];

#[cfg(not(miri))]
#[global_allocator]
static FERROC: Ferroc = Ferroc;

fn main() {
    println!("ferroc: {:?}", do_bench(&Ferroc));
    #[cfg(not(any(feature = "track-valgrind", miri)))]
    println!("system: {:?}", do_bench(&std::alloc::System));
}

fn do_bench<A: GlobalAlloc + Sync>(a: &A) -> Duration {
    if THREAD_COUNT == 1 {
        return bench_one(a);
    }

    thread::scope(|s| {
        let threads: Vec<_> = (0..THREAD_COUNT)
            .map(|_| s.spawn(|| bench_one(a)))
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .sum::<Duration>()
            / THREAD_COUNT.try_into().unwrap()
    })
}

struct BenchArg {
    size: usize,
    count: usize,
}

fn bench_one<A: GlobalAlloc>(alloc: &A) -> Duration {
    let mut memory: Vec<_> = iter::repeat_with(|| Allocation::UNINIT)
        .take(ROUND)
        .collect();
    let mut index = 0;
    let mut save_start = ROUND;
    let mut save_end = ROUND;

    let start = Instant::now();

    for _ in 0..ROUND {
        for &BenchArg { size, count } in BENCH_ARGS {
            #[cfg(miri)]
            let count = (count / 100).max(1);
            for _ in 0..count {
                memory[index] = unsafe { allocate_one(size, alloc) };
                index += 1;

                if index == save_start {
                    index = save_end;
                }

                if index == ROUND {
                    index = 0;

                    save_start = save_end;
                    if save_start >= ROUND {
                        save_start = 0;
                    }
                    save_end = save_start + ROUND / 5;
                    if save_end > ROUND {
                        save_end = ROUND;
                    }

                    memory[..save_start]
                        .iter_mut()
                        .for_each(|a| a.deallocate(alloc));
                    memory[save_end..]
                        .iter_mut()
                        .for_each(|a| a.deallocate(alloc));

                    if index == save_start {
                        index = save_end;
                    }
                }
            }
        }
    }

    memory[..index].iter_mut().for_each(|a| a.deallocate(alloc));
    if index < save_start {
        memory[save_start..save_end]
            .iter_mut()
            .for_each(|a| a.deallocate(alloc));
    }

    start.elapsed()
}

unsafe fn allocate_one<A: GlobalAlloc>(size: usize, a: &A) -> Allocation<A> {
    let layout = Layout::from_size_align(size, 32).unwrap();
    let ptr = NonNull::new(a.alloc(layout)).expect("out of memory");

    ptr.cast::<u8>().as_ptr().write(0);
    ptr.cast::<u8>().as_ptr().add(size - 1).write(0);

    Allocation {
        ptr: Some(ptr.cast()),
        layout,
        marker: PhantomData,
    }
}

struct Allocation<A> {
    ptr: Option<NonNull<u8>>,
    layout: Layout,
    marker: PhantomData<A>,
}

impl<A: GlobalAlloc> Allocation<A> {
    const UNINIT: Self = Allocation {
        ptr: None,
        layout: Layout::new::<()>(),
        marker: PhantomData,
    };

    fn deallocate(&mut self, a: &A) {
        if let Some(ptr) = self.ptr.take() {
            unsafe { a.dealloc(ptr.as_ptr(), self.layout) };
        }
    }
}
