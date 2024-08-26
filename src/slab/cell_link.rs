use core::cell::Cell;

pub trait CellLinked<'a>: PartialEq {
    fn link(&'a self) -> &'a CellLink<'a, Self>;
}

pub struct CellLink<'a, T: 'a + ?Sized> {
    linked_to: Cell<usize>,
    prev: Cell<Option<&'a T>>,
    next: Cell<Option<&'a T>>,
}

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
            linked_to: Cell::new(0),
            prev: Cell::new(None),
            next: Cell::new(None),
        }
    }
}

impl<'a, T> CellLink<'a, T> {
    pub fn is_linked(&self) -> bool {
        self.linked_to.get() != 0
    }
}

impl<'a, T> Default for CellLink<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CellList<'a, T: 'a + ?Sized> {
    head: Cell<Option<&'a T>>,
    tail: Cell<Option<&'a T>>,
    len: Cell<usize>,
}

impl<'a, T: CellLinked<'a>> CellList<'a, T> {
    #[allow(clippy::declare_interior_mutable_const)]
    pub const DEFAULT: Self = Self::new();

    pub const fn new() -> Self {
        CellList {
            head: Cell::new(None),
            tail: Cell::new(None),
            len: Cell::new(0),
        }
    }

    pub fn push(&self, value: &'a T) {
        debug_assert!(!value.link().is_linked());
        value.link().linked_to.set((self as *const Self).addr());

        let next = self.head.take();
        value.link().next.set(next);
        match next {
            Some(next) => next.link().prev.set(Some(value)),
            None => self.tail.set(Some(value)),
        }
        self.head.set(Some(value));
        self.len.set(self.len.get() + 1);
    }

    pub fn pop(&self) -> Option<&'a T> {
        self.head.take().inspect(|value| {
            self.len.set(self.len.get() - 1);
            let next = value.link().next.take();
            match next {
                Some(next) => next.link().prev.set(None),
                None => self.tail.set(None),
            }
            self.head.set(next);
            value.link().linked_to.set(0);
        })
    }

    pub fn contains(&self, value: &'a T) -> bool {
        value.link().linked_to.get() == (self as *const Self).addr()
    }

    pub fn remove(&self, value: &'a T) -> bool {
        if !self.contains(value) {
            return false;
        }

        self.len.set(self.len.get() - 1);
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
        value.link().linked_to.set(0);
        true
    }

    pub fn current(&self) -> Option<&'a T> {
        self.head.get()
    }

    pub fn len(&self) -> usize {
        self.len.get()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Iter<'a, T> {
        Iter(self.head.get())
    }

    pub fn cursor_head<'list>(&'list self) -> CellCursor<'a, 'list, T> {
        CellCursor { list: self, cur: self.head.get() }
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
            let value = self.cur.take()?;
            self.cur = value.link().next.get();
            if (self.pred)(value) {
                let _ret = self.list.remove(value);
                debug_assert!(_ret);
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

pub struct CellCursor<'a, 'list, T: CellLinked<'a>> {
    list: &'list CellList<'a, T>,
    cur: Option<&'a T>,
}

impl<'a, 'list, T: CellLinked<'a>> CellCursor<'a, 'list, T> {
    pub fn move_next(&mut self) -> bool {
        let next = self.cur.take().and_then(|cur| cur.link().next.get());
        self.cur = next;
        next.is_some()
    }

    pub fn get(&self) -> Option<&'a T> {
        self.cur
    }

    pub fn remove(&mut self) -> Option<&'a T> {
        let cur = self.cur.take();
        let next = cur.and_then(|cur| cur.link().next.get());
        self.cur = next;
        if let Some(cur) = cur {
            let _ret = self.list.remove(cur);
            debug_assert!(_ret);
        }
        cur
    }
}
