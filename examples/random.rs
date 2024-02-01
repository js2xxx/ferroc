use std::{
    alloc::{GlobalAlloc, Layout},
    iter,
    marker::PhantomData,
    ops::Range,
    ptr::NonNull,
    thread,
    time::{Duration, Instant},
};

use ferroc::Ferroc;

const ALL_ROUND: usize = 10000;
const THREAD_COUNT: usize = 24;
const ROUND: usize = ALL_ROUND / THREAD_COUNT;
const BLOCK_SIZES: Range<usize> = 1..1000;

fn main() {
    println!("ferroc: {:?}", do_bench(&Ferroc));
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

fn bench_one<A: GlobalAlloc>(alloc: &A) -> Duration {
    let mut memory: Vec<_> = iter::repeat_with(|| Allocation::UNINIT)
        .take(ROUND)
        .collect();
    let mut index = 0;
    let mut save_start = ROUND;
    let mut save_end = ROUND;

    let start = Instant::now();

    for _ in 0..ROUND {
        let mut size_base = BLOCK_SIZES.start;
        while size_base < BLOCK_SIZES.end {
            let mut size = size_base;
            while size > 0 {
                let iterations = match size {
                    ..=99 => 250,
                    100..=999 => 50,
                    1000..=9999 => 10,
                    _ => 1,
                };
                for _ in 0..iterations {
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
                size >>= 1;
            }

            size_base = size_base * 3 / 2 + 1;
        }
        memory[..index].iter_mut().for_each(|a| a.deallocate(alloc));
        if index < save_start {
            memory[save_start..save_end]
                .iter_mut()
                .for_each(|a| a.deallocate(alloc));
        }
    }

    start.elapsed()
}

unsafe fn allocate_one<A: GlobalAlloc>(size: usize, a: &A) -> Allocation<A> {
    let layout = Layout::array::<u8>(size).unwrap();
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
