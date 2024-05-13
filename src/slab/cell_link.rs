use core::{cell::Cell, ptr};

pub trait CellLinked<'a> {
    fn link(&'a self) -> &'a CellLink<'a, Self>;
}

pub struct CellLink<'a, T: ?Sized> {
    #[cfg(debug_assertions)]
    linked_to: Cell<usize>,
    prev: Cell<Option<&'a T>>,
    next: Cell<Option<&'a T>>,
}

#[cfg(debug_assertions)]
impl<'a, T: 'a + ?Sized> core::fmt::Debug for CellLink<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let linked_to = self.linked_to.get();
        if linked_to == 0 {
            write!(f, "CellLink (unlinked)")
        } else {
            write!(f, "CellLink ({:#x})", linked_to)
        }
    }
}

impl<'a, T> CellLink<'a, T> {
    pub const fn new() -> Self {
        CellLink {
            #[cfg(debug_assertions)]
            linked_to: Cell::new(0),
            prev: Cell::new(None),
            next: Cell::new(None),
        }
    }
}

impl<'a, T> CellLink<'a, T> {
    #[cfg(debug_assertions)]
    pub fn is_linked(&self) -> bool {
        self.linked_to.get() != 0
    }
}

impl<'a, T> Default for CellLink<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CellList<'a, T: ?Sized> {
    head: Cell<Option<&'a T>>,
    tail: Cell<Option<&'a T>>,
    #[cfg(debug_assertions)]
    len: Cell<usize>,
}

impl<'a, T: CellLinked<'a>> CellList<'a, T> {
    #[allow(clippy::declare_interior_mutable_const)]
    pub const DEFAULT: Self = Self::new();

    pub const fn new() -> Self {
        CellList {
            head: Cell::new(None),
            tail: Cell::new(None),
            #[cfg(debug_assertions)]
            len: Cell::new(0),
        }
    }

    pub fn push(&self, value: &'a T) {
        #[cfg(debug_assertions)]
        debug_assert!(!value.link().is_linked());
        #[cfg(debug_assertions)]
        value.link().linked_to.set(ptr::from_ref(self).addr());

        // INVARIANT: A link's `get` must be paired with a `set`, otherwise it must be
        // replaced with a `take`.
        let next = self.head.get();
        value.link().next.set(next);
        match next {
            Some(next) => next.link().prev.set(Some(value)),
            None => self.tail.set(Some(value)),
        }
        self.head.set(Some(value));

        #[cfg(debug_assertions)]
        self.len.set(self.len.get() + 1);
    }

    pub fn pop(&self) -> Option<&'a T> {
        let value = self.head.get()?;
        #[cfg(debug_assertions)]
        self.len.set(self.len.get() - 1);

        // INVARIANT: A link's `get` must be paired with a `set`, otherwise it must be
        // replaced with a `take`.
        let next = value.link().next.take();
        match next {
            Some(next) => next.link().prev.set(None),
            None => self.tail.set(None),
        }
        self.head.set(next);

        #[cfg(debug_assertions)]
        value.link().linked_to.set(0);
        Some(value)
    }

    #[cfg(debug_assertions)]
    pub fn contains(&self, value: &'a T) -> bool {
        value.link().linked_to.get() == ptr::from_ref(self).addr()
    }

    pub fn remove(&self, value: &'a T) {
        #[cfg(debug_assertions)]
        debug_assert!(self.contains(value));
        #[cfg(debug_assertions)]
        self.len.set(self.len.get() - 1);

        // INVARIANT: A link's `get` must be paired with a `set`, otherwise it must be
        // replaced with a `take`.
        let prev = value.link().prev.take();
        let next = value.link().next.take();
        match prev {
            Some(prev) => prev.link().next.set(next),
            None => self.head.set(next),
        }
        match next {
            Some(next) => next.link().prev.set(prev),
            None => self.tail.set(prev),
        }
        #[cfg(debug_assertions)]
        value.link().linked_to.set(0);
    }

    pub fn requeue_to(&self, value: &'a T, other: &Self) {
        #[cfg(debug_assertions)]
        debug_assert!(self.contains(value));

        #[cfg(debug_assertions)]
        self.len.set(self.len.get() - 1);
        #[cfg(debug_assertions)]
        other.len.set(other.len.get() + 1);
        #[cfg(debug_assertions)]
        value.link().linked_to.set(ptr::from_ref(other).addr());

        // INVARIANT: A link's `get` must be paired with a `set`, otherwise it must be
        // replaced with a `take`.
        let new_next = other.head.get();

        let last_prev = value.link().prev.take();
        let last_next = value.link().next.replace(new_next);

        match last_prev {
            Some(prev) => prev.link().next.set(last_next),
            None => self.head.set(last_next),
        }
        match last_next {
            Some(next) => next.link().prev.set(last_prev),
            None => self.tail.set(last_prev),
        }
        match new_next {
            Some(next) => next.link().prev.set(Some(value)),
            None => other.tail.set(Some(value)),
        }

        other.head.set(Some(value));
    }

    pub fn current(&self) -> Option<&'a T> {
        self.head.get()
    }

    #[cfg(debug_assertions)]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    #[cfg(debug_assertions)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn has_sole_member(&self) -> bool {
        let head_addr = self
            .head
            .get()
            .map(|x| ptr::from_ref(x).addr())
            .unwrap_or(0);
        let tail_addr = self
            .tail
            .get()
            .map(|x| ptr::from_ref(x).addr())
            .unwrap_or(0);
        head_addr == tail_addr
    }

    pub fn iter(&self) -> Iter<'a, T> {
        Iter(self.head.get())
    }

    pub fn cursor_head(&self) -> CellCursor<'a, T> {
        CellCursor { cur: self.head.get() }
    }

    /// # Arguments
    ///
    /// - `pred`: `true` indicates that the current element should be removed,
    ///   vice versa.
    pub fn drain<'list, F>(&'list self, pred: F) -> Drain<'a, 'list, T, F>
    where
        F: FnMut(&'a T) -> bool,
    {
        Drain {
            list: self,
            cur: self.head.get(),
            pred,
        }
    }
}

impl<'a, T: CellLinked<'a>> Default for CellList<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: CellLinked<'a>> IntoIterator for &CellList<'a, T> {
    type IntoIter = Iter<'a, T>;

    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Iter<'a, T>(Option<&'a T>);

impl<'a, T: CellLinked<'a>> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.take().inspect(|obj| self.0 = obj.link().next.get())
    }
}

pub struct Drain<'a, 'list, T: CellLinked<'a>, F: FnMut(&'a T) -> bool> {
    list: &'list CellList<'a, T>,
    cur: Option<&'a T>,
    pred: F,
}

impl<'a, 'list, T: CellLinked<'a>, F: FnMut(&'a T) -> bool> Iterator for Drain<'a, 'list, T, F> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let value = self.cur?;
            self.cur = value.link().next.get();
            if (self.pred)(value) {
                self.list.remove(value);
                break Some(value);
            }
        }
    }
}

impl<'a, 'list, T: CellLinked<'a>, F: FnMut(&'a T) -> bool> Drop for Drain<'a, 'list, T, F> {
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

pub struct CellCursor<'a, T: CellLinked<'a>> {
    cur: Option<&'a T>,
}

impl<'a, T: CellLinked<'a>> CellCursor<'a, T> {
    pub fn move_next(&mut self) {
        self.cur = self.cur.and_then(|cur| cur.link().next.get());
    }

    pub fn get(&self) -> Option<&'a T> {
        self.cur
    }
}
