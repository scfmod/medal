use crate::{type_system::Infer, SideEffects, Traverse, Type, TypeSystem};
use by_address::ByAddress;
use derive_more::From;
use enum_dispatch::enum_dispatch;
use nohash_hasher::NoHashHasher;
use parking_lot::Mutex;
use std::{
    fmt::{self, Display},
    hash::{Hash, Hasher},
};
use triomphe::Arc;

#[derive(Debug, Default, From, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Local(pub Option<String>);

impl Local {
    pub fn new(name: Option<String>) -> Self {
        Self(name)
    }
}

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            Some(name) => write!(f, "{}", name),
            None => write!(f, "UNNAMED_LOCAL"),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RcLocal(pub ByAddress<Arc<Mutex<Local>>>);

impl Infer for RcLocal {
    fn infer<'a: 'b, 'b>(&'a mut self, system: &mut TypeSystem<'b>) -> Type {
        system.type_of(self).clone()
    }
}

impl Display for RcLocal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 .0.lock().0 {
            Some(name) => write!(f, "{}", name),
            None => {
                let mut hasher = NoHashHasher::<u8>::default();
                self.hash(&mut hasher);
                write!(f, "UNNAMED_{}", hasher.finish())
            }
        }
    }
}

impl SideEffects for RcLocal {}

impl Traverse for RcLocal {}

impl RcLocal {
    pub fn new(local: Local) -> Self {
        Self(ByAddress(Arc::new(Mutex::new(local))))
    }

    /// Create a new RcLocal that copies the name from an existing local.
    /// This is useful for SSA construction where we want to preserve
    /// debug names through variable renaming.
    pub fn clone_with_name(other: &RcLocal) -> Self {
        let name = other.0 .0.lock().0.clone();
        Self::new(Local::new(name))
    }

    /// Get the name of this local, if any
    pub fn name(&self) -> Option<String> {
        self.0 .0.lock().0.clone()
    }

    /// Set the name of this local, but only if it doesn't already have one
    pub fn set_name_if_unnamed(&self, name: Option<String>) {
        let mut lock = self.0 .0.lock();
        if lock.0.is_none() {
            lock.0 = name;
        }
    }
}

impl LocalRw for RcLocal {
    fn values_read(&self) -> Vec<&RcLocal> {
        vec![self]
    }

    fn values_read_mut(&mut self) -> Vec<&mut RcLocal> {
        vec![self]
    }
}

#[enum_dispatch]
pub trait LocalRw {
    fn values_read(&self) -> Vec<&RcLocal> {
        Vec::new()
    }

    fn values_read_mut(&mut self) -> Vec<&mut RcLocal> {
        Vec::new()
    }

    fn values_written(&self) -> Vec<&RcLocal> {
        Vec::new()
    }

    fn values_written_mut(&mut self) -> Vec<&mut RcLocal> {
        Vec::new()
    }

    fn values(&self) -> Vec<&RcLocal> {
        self.values_read()
            .into_iter()
            .chain(self.values_written())
            .collect()
    }

    fn replace_values_read(&mut self, old: &RcLocal, new: &RcLocal) {
        for value in self.values_read_mut() {
            if value == old {
                *value = new.clone();
            }
        }
    }

    fn replace_values_written(&mut self, old: &RcLocal, new: &RcLocal) {
        for value in self.values_written_mut() {
            if value == old {
                *value = new.clone();
            }
        }
    }

    fn replace_values(&mut self, old: &RcLocal, new: &RcLocal) {
        self.replace_values_read(old, new);
        self.replace_values_written(old, new);
    }
}
