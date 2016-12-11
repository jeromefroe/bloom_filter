// MIT License

// Copyright (c) 2016 Jerome Froelich

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! An implementation of a bloom filter as as described in
//! [Space/Time Trade-offs in Hash Coding with Allowable Errors] (http://dmod.eu/deca/ft_gateway.cfm.pdf).
//!
//! ## Example
//!
//! ``` rust,no_run
//!
//! fn main() {
//! }
//! ```

extern crate bit_vec;
extern crate rand;
extern crate siphasher;

use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use bit_vec::BitVec;
use siphasher::sip::SipHasher24;


#[derive(Debug)]
pub enum BloomError {
    NoParameterSet,
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BloomError::NoParameterSet => {
                write!(f,
                       "Must set either the desired size or false positive ratio to create a \
                        bloom filter")
            }
        }
    }
}

impl Error for BloomError {
    fn description(&self) -> &str {
        match *self {
            BloomError::NoParameterSet => {
                "Must set either the desired size or false positive ratio to create a \
                        bloom filter"
            }
        }
    }
}

#[derive(Debug)]
enum BloomParameter {
    Empty,
    Size(u64), // number of bits in the bloom filter
    FPR(f64), // false positive rate
}

#[derive(Debug)]
pub struct BloomBuilder {
    elements: u64,
    parameter: BloomParameter,
}

impl BloomBuilder {
    pub fn new(elements: u64) -> BloomBuilder {
        BloomBuilder {
            elements: elements,
            parameter: BloomParameter::Empty,
        }
    }

    pub fn with_size(mut self, size: u64) -> BloomBuilder {
        self.parameter = BloomParameter::Size(size);
        self
    }

    pub fn with_fpr(mut self, p: f64) -> BloomBuilder {
        self.parameter = BloomParameter::FPR(p);
        self
    }

    pub fn finish<T>(&self) -> Result<Bloom<T>, BloomError>
        where T: Hash
    {
        match self.parameter {
            BloomParameter::Empty => Err(BloomError::NoParameterSet),
            BloomParameter::Size(size) => {
                let hash_count = BloomBuilder::optimal_hash_count(self.elements, size);
                let bloom = Bloom::new(size, hash_count);
                Ok(bloom)
            }
            BloomParameter::FPR(p) => {
                let min_size = BloomBuilder::min_size(self.elements, p);
                let hash_count = BloomBuilder::optimal_hash_count(self.elements, min_size);
                let bloom: Bloom<T> = Bloom::new(min_size, hash_count);
                Ok(bloom)
            }
        }
    }

    fn min_size(num_elements: u64, p: f64) -> u64 {
        let n = num_elements as f64;
        // m = -1 * n * ln(p) / (ln(2) ^ 2)
        let m = -1f64 * n * p.ln() / (2f64.ln().powf(2f64));
        m.ceil() as u64
    }

    fn optimal_hash_count(num_elements: u64, num_bits: u64) -> u32 {
        let m = num_bits as f64;
        let n = num_elements as f64;
        // k = ln(2) * m / n
        let k = 2f64.ln() * m / n;
        k.ceil() as u32
    }
}

#[derive(Debug)]
pub struct Bloom<T: Hash> {
    bits: BitVec,

    // the number of hash functions to compute when inserting and looking up elements
    k: u32,

    // we only need two independent hash functions because we can use double hashing to
    // create new independent hash functions from those two
    hashers: [SipHasher24; 2],

    marker: PhantomData<T>,
}

impl<T: Hash> Bloom<T> {
    fn new(size: u64, k: u32) -> Self {

        Bloom {
            bits: BitVec::from_elem(size as usize, false),
            k: k,
            hashers: [Bloom::<T>::get_hasher(), Bloom::<T>::get_hasher()],
            marker: PhantomData,
        }
    }

    fn get_hasher() -> SipHasher24 {
        let mut rng = rand::thread_rng();
        SipHasher24::new_with_keys(rand::Rand::rand(&mut rng), rand::Rand::rand(&mut rng))
    }

    pub fn insert(&mut self, key: T) {
        // we need to save the results of the first two hashes for when we need to get more
        // than two hashes since we use double hashing uses those hashes to compute further hashes
        let mut hashes = [0u64, 0u64];
        for i in 0..self.k {
            let index = self.get_hash(&mut hashes, &key, i) % self.bits.len();
            self.bits.set(index, true);
        }
    }

    pub fn lookup(&self, item: T) -> bool {
        let mut hashes = [0u64, 0u64];
        for i in 0..self.k {
            let index = self.get_hash(&mut hashes, &item, i) % self.bits.len();
            if !self.bits[index] {
                return false;
            }
        }

        true
    }

    pub fn lookup_and_insert(&mut self, key: T) -> bool {
        let mut hashes = [0u64, 0u64];
        let mut found = true;
        for i in 0..self.k {
            let index = self.get_hash(&mut hashes, &key, i) % self.bits.len();
            if !self.bits[index] {
                found = false;
                self.bits.set(index, true);
            }
        }

        found
    }

    fn get_hash(&self, hashes: &mut [u64; 2], key: &T, i: u32) -> usize {
        if i < 2 {
            let hasher = &mut self.hashers[i as usize].clone();
            key.hash(hasher);
            let hash = hasher.finish();
            hashes[i as usize] = hash;
            hash as usize
        } else {
            // use double hashing to get any additional hashes
            hashes[0]
                .wrapping_add((i as u64).wrapping_mul(hashes[1]) %
                              self.bits.len() as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BloomBuilder;

    #[test]
    fn test_with_size() {
        let size = 2u64.pow(20);
        let b = BloomBuilder::new(size / 2).with_size(size).finish::<i64>().unwrap();
        assert_eq!(b.bits.len() as u64, size);
        assert_eq!(b.k, 2);
    }

    #[test]
    fn test_with_fpr() {
        let elements = 2u64.pow(20);
        let fpr = 0.01;
        let b = BloomBuilder::new(elements).with_fpr(fpr).finish::<i64>().unwrap();
        println!("{:?}", b.bits.len());
        assert_eq!(b.k, 7);
    }

    #[test]
    fn test_operations() {
        let elements = 2u64.pow(20);
        let fpr = 0.01;
        let mut b = BloomBuilder::new(elements).with_fpr(fpr).finish::<i64>().unwrap();

        assert!(!b.lookup(3));
        assert!(!b.lookup(23));
        assert!(!b.lookup(51));

        b.insert(3);
        b.insert(23);
        b.insert(51);

        assert!(b.lookup(3));
        assert!(b.lookup(23));
        assert!(b.lookup(51));

        assert!(b.lookup_and_insert(3));
        assert!(b.lookup_and_insert(23));
        assert!(b.lookup_and_insert(51));
    }
}
