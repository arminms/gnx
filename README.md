<div align="center">
  <a href="https://github.com/arminms/gnx">
    <img width="256" heigth="256" src="doc/images/gnx_features.png">
  </a>
  <h1>GNX</h1>
</div>

[![Build and Test (Linux/macOS/Windows)](https://github.com/arminms/gnx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/arminms/gnx/actions/workflows/cmake-multi-platform.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Codespaces-Open-yellowgreen?logo=github&logoColor=lightgray)](https://codespaces.new/arminms/gnx)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/gnx/HEAD?urlpath=%2Fdoc%2Ftree%2Fsandbox.ipynb)

`GNX` (pronounced *jeenks*) is a *Modern C++* library and a command line toolkit (`gnx`) that designed to work in *Jupyter Notebooks* with [Xeus-Cling](https://github.com/jupyter-xeus/xeus-cling) kernel for rapid-prototyping of all sorts of biosequence analysis. Built on *C++20* features — e.g. [*Concepts*](https://en.cppreference.com/cpp/language/constraints) and [*Ranges*](https://en.cppreference.com/cpp/ranges) — GNX is designed for zero-copy operations, [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)/parallel accelerations, and seamless portability between [OpenMP](), [CUDA](https://en.wikipedia.org/wiki/CUDA), and [ROCm](https://en.wikipedia.org/wiki/ROCm) backends.

To start off working in a guided sandbox, use the command below or click on `launch binder` badge above:

```
docker run -p 8888:8888 -it --rm asobhani/gnx
```
