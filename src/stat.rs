use crate::heap::OBJ_SIZE_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stat {
    pub slabs: usize,
    pub shards: usize,

    // pub direct_size: usize,
    pub normal_size: usize,
    pub huge_size: usize,

    // pub direct_count: usize,
    pub normal_count: [usize; OBJ_SIZE_COUNT],
    pub huge_count: usize,

    pub free_shards: usize,

    pub abandoned_slabs: usize,
    pub abandoned_shards: usize,
}

impl Default for Stat {
    fn default() -> Self {
        Self::INIT
    }
}

impl Stat {
    pub const INIT: Stat = Stat {
        slabs: 0,
        shards: 0,
        normal_size: 0,
        huge_size: 0,
        // direct_size: 0,
        normal_count: [0; OBJ_SIZE_COUNT],
        huge_count: 0,
        // direct_count: 0,
        free_shards: 0,
        abandoned_slabs: 0,
        abandoned_shards: 0,
    };

    pub fn assert_clean(&self) {
        assert_eq!(
            self.slabs,
            self.abandoned_slabs,
            "{} slab(s) is(are) probably leaked",
            self.slabs - self.abandoned_slabs
        );
        assert_eq!(
            self.shards,
            self.abandoned_shards,
            "{} shard(s) is(are) probably leaked",
            self.shards - self.abandoned_shards
        );
    }
}
