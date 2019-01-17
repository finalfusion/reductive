# Reductive

## Training of optimized product quantizers

Training of *optimized* product quantizers requires a LAPACK
implementation. For this reason, training of the `OPQ` and
`GaussianOPQ` quantized is feature-gated by the `opq-train` feature.
Without the the `opq-train` feature, you can train the `PQ` quantizer
and use pre-trained quantizers.

If you use the `opq-train` feature, you also have to select a
BLAS/LAPACK implementation. The supported implementations are:

* OpenBLAS (feature: `openblas`)
* Netlib (feature: `netlib`)
* Intel MKL (feature: `intel-mk;`)

There is a feature for macOS Accelerate (`accelerate`). However,
Accelerate does not currently provide the necessary LAPACK
routines. This feature is present in case Accelerate adds the
necessary routines.

The `opq-train` feature and a backend can be enabled as follows:

~~~toml
[dependencies]
reductive = { version = "0.1", features = ["opq-train", "openblas"] }
~~~

### Running tests

To run *all* tests, enable the `opq-train` feature and specify the
BLAS/LAPACK implementation:

~~~shell
$ cargo test --verbose --features "opq-train openblas"
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
