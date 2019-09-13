# Reductive

## Training of optimized product quantizers

Training of *optimized* product quantizers requires a LAPACK
implementation. For this reason, training of the `OPQ` and
`GaussianOPQ` quantizers is feature-gated by the `opq-train` feature.
`opq-train` is automatically enabled by selecting a BLAS/LAPACK
implementation. The supported implementations are:

* OpenBLAS (feature: `openblas`)
* Netlib (feature: `netlib`)
* Intel MKL (feature: `intel-mk;`)

A backend can be selected as follows:

~~~toml
[dependencies]
reductive = { version = "0.3", features = ["openblas"] }
~~~

### Running tests

To run *all* tests, specify the BLAS/LAPACK implementation:

~~~shell
$ cargo test --verbose --features "openblas"
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
