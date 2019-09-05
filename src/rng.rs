use rand::{RngCore, SeedableRng};

/// RNG that reseeds on clone.
///
/// This is a wrapper struct for RNGs implementing the `RngCore`
/// trait.  It adds the following simple behavior: when a
/// `ReseedOnCloneRng` is cloned, the clone is constructed using fresh
/// entropy. This assures that the state of the clone is not related
/// to the cloned RNG.
///
/// The `rand` crate provides similar behavior in the `ReseedingRng`
/// struct. However, `ReseedingRng` requires that the RNG is
/// `BlockRngCore`.
pub struct ReseedOnCloneRng<R>(pub R)
where
    R: RngCore + SeedableRng;

impl<R> RngCore for ReseedOnCloneRng<R>
where
    R: RngCore + SeedableRng,
{
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.0.try_fill_bytes(dest)
    }
}

impl<R> Clone for ReseedOnCloneRng<R>
where
    R: RngCore + SeedableRng,
{
    fn clone(&self) -> Self {
        ReseedOnCloneRng(R::from_entropy())
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_core::{self, impls, le, RngCore};

    use super::ReseedOnCloneRng;

    #[derive(Clone)]
    struct BogusRng(pub u64);

    impl RngCore for BogusRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            impls::fill_bytes_via_next(self, dest)
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            Ok(self.fill_bytes(dest))
        }
    }

    impl SeedableRng for BogusRng {
        type Seed = [u8; 8];

        fn from_seed(seed: Self::Seed) -> Self {
            let mut state = [0u64; 1];
            le::read_u64_into(&seed, &mut state);
            BogusRng(state[0])
        }
    }

    #[test]
    fn reseed_on_clone_rng() {
        let bogus_rng = BogusRng::from_entropy();
        let bogus_rng_clone = bogus_rng.clone();
        assert_eq!(bogus_rng.0, bogus_rng_clone.0);

        let reseed = ReseedOnCloneRng(bogus_rng);
        let reseed_clone = reseed.clone();
        // One in 2^64 probability of collision given good entropy source.
        assert_ne!((reseed.0).0, (reseed_clone.0).0);
    }
}
