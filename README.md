# Reductive

## Training of optimized product quantizers

Training of *optimized* product quantizers requires a LAPACK implementation. For
this reason, training of the `Opq` and `GaussianOpq` quantizers is feature-gated
by the `opq-train` feature.  This feature must be enabled if you want to use
`Opq` or `GaussianOpq`:

~~~toml
[dependencies]
reductive = { version = "0.7", features = ["opq-train"] }
~~~

This also requires that a crate that links a LAPACK library is added as a
dependency, e.g. `accelerate-src`, `intel-mkl-src`, `openblas-src`, or
`netlib-src`.

### Running tests

#### Linux

You can run all tests on Linux, including tests for optimized product
quantizers, using the `intel-mkl-test` feature:

~~~shell
$ cargo test --features intel-mkl-test
~~~

#### macOS

All tests can be run on macOS with the `accelerate-test` feature:

~~~shell
$ cargo test --features accelerate-test
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
