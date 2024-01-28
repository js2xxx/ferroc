use core::{
    mem,
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering::*},
};

const BITS: u32 = usize::BITS;

#[derive(Debug)]
#[repr(transparent)]
pub struct Bitmap([AtomicUsize]);

#[derive(Debug)]
enum Ranges {
    Infield(usize, Range<u32>),
    Across {
        first: (usize, u32),
        mid: Range<usize>,
        last: (usize, u32),
    },
}

impl Bitmap {
    pub fn new(storage: &[AtomicUsize]) -> &Self {
        // SAFETY: `Bitmap` is `repr(transparent)`.
        unsafe { mem::transmute(storage) }
    }

    // pub fn empty<'a>() -> &'a Self {
    //     Self::new(&[])
    // }

    pub fn len(&self) -> u32 {
        self.0.len() as u32 * BITS
    }

    // pub fn is_empty(&self) -> bool {
    //     self.0.is_empty()
    // }

    fn ranges(range: Range<u32>) -> Ranges {
        let (first_sto, first_bit) = ((range.start / BITS) as usize, range.start % BITS);
        let (last_sto, last_bit) = ((range.end / BITS) as usize, range.end % BITS);

        if first_sto == last_sto {
            return Ranges::Infield(first_sto, first_bit..last_bit);
        }

        let mid_start = first_sto + 1;
        let mid_end = last_sto;

        Ranges::Across {
            first: (first_sto, first_bit),
            mid: mid_start..mid_end,
            last: (last_sto, last_bit),
        }
    }

    fn loop_at(&self, start: usize) -> impl Iterator<Item = (usize, &AtomicUsize)> {
        let enumerate = self.0.iter().enumerate();
        enumerate.cycle().skip(start).take(self.0.len())
    }

    pub fn allocate_infield(&self, start: usize, count: u32) -> Option<(usize, u32)> {
        self.loop_at(start)
            .find_map(|(idx, sto)| Some((idx, sto.allocate(count)?)))
    }

    /// Tries to find `count` zero bits starting from storage index `start`
    /// across storage units, and set them, at most 3 times.
    fn allocate_across_in(&self, start: usize, count: u32) -> Option<u32> {
        let start_sto = &self.0[start];

        let mut cur = self.0[start].load(Relaxed);
        let mut retries = 0;
        const MAX_RETRIES: usize = 3;
        loop {
            // Check leading zero bits:
            let first = cur.leading_zeros();
            match first {
                // No leading zeros, fail;
                0 => return None,
                // Sufficient zero bits, allocate them within the boundary;
                _ if first >= count => return start_sto.allocate(count),
                // Capacity not enough, fail;
                _ if (count - first).div_ceil(BITS) as usize >= self.0.len() - start => {
                    return None;
                }
                _ => {}
            }

            let start_bit = BITS - first;
            let end_idx = start as u32 * BITS + start_bit + count;
            let (end, end_bit) = ((end_idx / BITS) as usize, end_idx % BITS);

            let mid_sto = &self.0[(start + 1)..end];
            let end_sto = (end_bit != 0).then_some(&self.0[end]);

            // Check all the following bits.
            if mid_sto.iter().any(|sto| sto.load(Relaxed) != 0)
                || end_sto.map_or(false, |sto| sto.load(Relaxed) & !(!0 << end_bit) != 0)
            {
                return None;
            }

            let start_mask = !0 << start_bit;
            let end_mask = !(!0 << end_bit);

            // Tries the CAS all the checked bits; goes out of this block if CAS fails.
            let err_pos = 'trial: {
                // CAS the starting bits.
                cur = start_sto.load(Relaxed);
                loop {
                    if cur & start_mask != 0 {
                        break 'trial start;
                    }
                    match start_sto.compare_exchange(cur, cur | start_mask, AcqRel, Acquire) {
                        Ok(_) => break,
                        Err(e) => cur = e,
                    }
                }

                // CAS the middle bits.
                if let Some(pos) = mid_sto
                    .iter()
                    .position(|sto| sto.compare_exchange(0, !0, AcqRel, Acquire).is_err())
                {
                    break 'trial start + 1 + pos;
                }

                // CAS the end bits, if any.
                if let Some(end_sto) = end_sto {
                    cur = end_sto.load(Relaxed);
                    loop {
                        if cur & end_mask != 0 {
                            break 'trial end;
                        }
                        match end_sto.compare_exchange(cur, cur | end_mask, AcqRel, Acquire) {
                            Ok(_) => break,
                            Err(e) => cur = e,
                        }
                    }
                }

                return Some(start_bit);
            };

            // Roll back possibly done middle bits.
            let mid_done = self.0[(start + 1)..err_pos].iter().rev();
            mid_done.for_each(|sto| sto.store(0, Release));

            // Roll back possibly done start bits.
            if err_pos > start {
                loop {
                    match start_sto.compare_exchange(cur, cur & !start_mask, AcqRel, Acquire) {
                        Ok(_) => break,
                        Err(e) => cur = e,
                    }
                }
            }

            retries += 1;
            if retries >= MAX_RETRIES {
                return None;
            }
        }
    }

    pub fn allocate(&self, start: usize, count: u32) -> Option<(usize, u32)> {
        if count <= 2 {
            return self.allocate_infield(start, count);
        }

        let can_allocate_infield = count <= BITS;
        self.loop_at(start).find_map(|(idx, sto)| {
            can_allocate_infield
                .then(|| sto.allocate(count))
                .or_else(|| Some(self.allocate_across_in(start, count)))
                .and_then(|bit| Some((idx, bit?)))
        })
    }

    fn walk_storage(
        &self,
        range: Range<u32>,
        mut f: impl FnMut(&AtomicUsize, Range<u32>) -> (bool, bool),
    ) -> (bool, bool) {
        match Self::ranges(range) {
            Ranges::Infield(sto, range) => f(&self.0[sto], range),
            Ranges::Across {
                first: (first_sto, first_bit),
                mid,
                last: (last_sto, last_bit),
            } => {
                let (fz, fo) = f(&self.0[first_sto], first_bit..BITS);
                let (mz, mo) = self.0[mid].iter().fold((false, false), |(mz, mo), sto| {
                    let (z, o) = f(sto, 0..BITS);
                    (mz || z, mo || o)
                });
                let (lz, lo) = (last_bit > 0)
                    .then(|| f(&self.0[last_sto], 0..last_bit))
                    .unwrap_or_default();
                (fz || mz || lz, fo || mo || lo)
            }
        }
    }

    pub fn set<const VALUE: bool>(&self, range: Range<u32>) -> (bool, bool) {
        self.walk_storage(range, StorageExt::set::<VALUE>)
    }

    // pub fn get(&self, range: Range<u32>) -> (bool, bool) {
    //     self.walk_storage(range, StorageExt::get)
    // }
}

fn mask(range: Range<u32>) -> usize {
    debug_assert!(range.start < usize::BITS);
    debug_assert!(range.end <= usize::BITS);

    ((!0usize).wrapping_shl(range.start)) & !((!0usize).checked_shl(range.end).unwrap_or(0))
}

trait StorageExt {
    fn get(&self, range: Range<u32>) -> (bool, bool);
    fn allocate(&self, count: u32) -> Option<u32>;

    fn set<const VALUE: bool>(&self, range: Range<u32>) -> (bool, bool);
}

impl StorageExt for AtomicUsize {
    fn allocate(&self, count: u32) -> Option<u32> {
        let mask = !(!0 << count);
        let end = BITS - count;

        let mut cur = self.load(Relaxed);
        loop {
            let mut bit = cur.trailing_ones();
            let bit = loop {
                if bit > end {
                    return None;
                }
                let test = cur & (mask << bit);
                if test == 0 {
                    break bit;
                }
                bit = BITS - test.leading_zeros();
            };
            match self.compare_exchange(cur, cur | (mask << bit), AcqRel, Acquire) {
                Ok(_) => break Some(bit),
                Err(c) => cur = c,
            }
        }
    }

    fn set<const VALUE: bool>(&self, range: Range<u32>) -> (bool, bool) {
        let mask = mask(range);
        let prev = match VALUE {
            true => self.fetch_or(mask, AcqRel),
            false => self.fetch_and(!mask, AcqRel),
        };
        (prev & mask != mask, prev & mask != 0)
    }

    fn get(&self, range: Range<u32>) -> (bool, bool) {
        let mask = mask(range);
        let value = self.load(Relaxed);
        (value & mask != mask, value & mask != 0)
    }
}
