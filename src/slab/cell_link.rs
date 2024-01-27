use core::{cell::Cell, ops::ControlFlow};

pub trait CellLinked: Sized + Copy {
    fn link(&self) -> &CellLink<Self>;
}

pub struct CellLink<T> {
    is_linked: Cell<bool>,
    prev: Cell<Option<T>>,
    next: Cell<Option<T>>,
}

impl<T> CellLink<T> {
    pub const fn new() -> Self {
        CellLink {
            is_linked: Cell::new(false),
            prev: Cell::new(None),
            next: Cell::new(None),
        }
    }
}

impl<T: Copy> CellLink<T> {
    pub fn is_linked(&self) -> bool {
        self.is_linked.get()
    }
}

impl<T> Default for CellLink<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CellList<T> {
    head: Cell<Option<T>>,
    tail: Cell<Option<T>>,
    len: Cell<usize>,
}

impl<T: CellLinked> CellList<T> {
    pub const DEFAULT: Self = Self::new();

    pub const fn new() -> Self {
        CellList {
            head: Cell::new(None),
            tail: Cell::new(None),
            len: Cell::new(0),
        }
    }

    pub fn push(&self, value: T) {
        debug_assert!(!value.link().is_linked());
        value.link().is_linked.set(true);

        let next = self.head.take();
        value.link().next.set(next);
        match next {
            Some(next) => next.link().prev.set(Some(value)),
            None => self.tail.set(Some(value)),
        }
        self.head.set(Some(value));
        self.len.set(self.len.get() + 1);
    }

    pub fn pop(&self) -> Option<T> {
        self.head.take().inspect(|value| {
            self.len.set(self.len.get() - 1);
            let next = value.link().next.take();
            match next {
                Some(next) => next.link().prev.set(None),
                None => self.tail.set(None),
            }
            self.head.set(next);
            value.link().is_linked.set(false);
        })
    }

    pub fn remove(&self, value: T) {
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
        value.link().is_linked.set(false);
    }

    pub fn current(&self) -> Option<T> {
        self.head.get()
    }

    pub fn len(&self) -> usize {
        self.len.get()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Iter<T> {
        Iter(self.head.get())
    }

    pub fn cursor_head(&self) -> CellCursor<T> {
        CellCursor { list: self, cur: self.head.get() }
    }

    /// # Arguments
    ///
    /// - `pred`: `true` indicates that the current element should be removed,
    ///   vice versa.
    pub fn drain<F>(&self, pred: F) -> Drain<T, F>
    where
        F: FnMut(T) -> bool,
    {
        Drain {
            list: self,
            cur: self.head.get(),
            pred,
        }
    }

    /// # Arguments
    ///
    /// - `pred`: `true` indicates the current element should be retained.
    pub fn retain<F, R>(&self, mut pred: F) -> Option<R>
    where
        F: FnMut(T) -> (bool, ControlFlow<R, ()>),
    {
        let mut cur = self.head.get();
        while let Some(node) = cur {
            let (remove, control_flow) = pred(node);
            match control_flow {
                ControlFlow::Continue(()) => cur = node.link().next.get(),
                ControlFlow::Break(r) => return Some(r),
            }
            if remove {
                self.remove(node);
            }
        }
        None
    }
}

impl<T: CellLinked> Default for CellList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CellLinked> IntoIterator for &CellList<T> {
    type IntoIter = Iter<T>;

    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Iter<T>(Option<T>);

impl<T: CellLinked> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.take().inspect(|obj| self.0 = obj.link().next.get())
    }
}

pub struct Drain<'a, T: CellLinked, F: FnMut(T) -> bool> {
    list: &'a CellList<T>,
    cur: Option<T>,
    pred: F,
}

impl<'a, T: CellLinked, F: FnMut(T) -> bool> Iterator for Drain<'a, T, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let value = self.cur.take()?;
            self.cur = value.link().next.get();
            if (self.pred)(value) {
                self.list.remove(value);
                break Some(value);
            }
        }
    }
}

impl<'a, T: CellLinked, F: FnMut(T) -> bool> Drop for Drain<'a, T, F> {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

pub struct CellCursor<'a, T: CellLinked> {
    list: &'a CellList<T>,
    cur: Option<T>,
}

impl<'a, T: CellLinked> CellCursor<'a, T> {
    pub fn move_next(&mut self) -> bool {
        let next = self.cur.take().and_then(|cur| cur.link().next.get());
        self.cur = next;
        next.is_some()
    }

    pub fn get(&self) -> Option<&T> {
        self.cur.as_ref()
    }

    pub fn remove(&mut self) -> Option<T> {
        let cur = self.cur.take();
        let next = cur.and_then(|cur| cur.link().next.get());
        self.cur = next;
        if let Some(cur) = cur {
            self.list.remove(cur);
        }
        cur
    }
}
