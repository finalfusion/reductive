# Reductive

## LAPACK

`reductive` requires a LAPACK implementation. Binary crates that use
`reductive` must specify the BLAS/LAPACK implementation to be
used. The following options are supported:

* OpenBLAS (feature: `openblas`)
* Netlib (feature: `netlib`)
* Intel MKL (feature: `intel-mk;`)

There is a feature for macOS Accelerate (`accelerate`). However,
Accelerate does not currently provide the necessary LAPACK
routines. This feature is present in case Accelerate adds the
necessary routines.

The backend can be selected as follows:

~~~toml
[dependencies]
reductive = { version = "0.1", features = ["openblas"] }
~~~

### Running tests

When running tests, specify the BLAS/LAPACK implementation as a feature:

~~~shell
$ cargo test --verbose --features openblas
~~~

### Multi-threaded OpenBLAS

`reductive` uses Rayon to parallelize quantizer training. However,
multi-threaded OpenBLAS is [known to
conflict](https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded)
with application threading. Is you use OpenBLAS, ensure that threading
is disabled, for instance by setting the number of threads to 1:

~~~shell
$ export OPENBLAS_NUM_THREADS=1
~~~
