# MPI + CUDA Parallel Computing Project

This project demonstrates the use of **MPI (Message Passing Interface)** and **CUDA (Compute Unified Device Architecture)** for parallel computing. It implements **Matrix Multiplication**, **Phonebook Searching**, and **Substring Matching** using MPI for distributed processing and CUDA for GPU acceleration.

---

## 📚 Theoretical Background

### What is MPI?

**MPI (Message Passing Interface)** is a standard for parallel programming used to allow processes to communicate and share data in distributed computing systems. It is commonly used in **multicore, multiprocessor, and cluster computing environments**. MPI enables the development of parallel programs that run efficiently across different nodes or processors.

- **Key MPI operations** include:
  - Point-to-point communication (e.g., `send`/`receive`)
  - Collective communication (e.g., `broadcast`, `gather`)
  - Synchronization mechanisms (e.g., barriers)

**Why is MPI used?**
- MPI allows the parallelization of computational tasks by dividing work into smaller units and distributing them across multiple processors or nodes. This significantly reduces the time for large-scale computations, enabling faster results in scientific computing, simulations, and more.

---

### What is CUDA?

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and programming model developed by NVIDIA. It allows software developers to use a **GPU (Graphics Processing Unit)** for general-purpose computing, accelerating performance-intensive tasks.

- **Key CUDA features:**
  - Parallel execution on hundreds or thousands of GPU cores
  - Efficient memory management with shared memory, global memory, and registers
  - CUDA supports C, C++, and Fortran for programming GPU applications

**Why is CUDA used?**
- CUDA accelerates computational tasks by offloading intensive calculations to the GPU, which is optimized for parallel data processing. This is particularly beneficial for tasks like **matrix operations, image processing, machine learning, and simulations** that can leverage parallelism.

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

1. **NVIDIA GPU**: A compatible GPU for CUDA execution.
2. **CUDA Toolkit**: [Download here](https://developer.nvidia.com/cuda-downloads) and install the appropriate version for your system.
3. **MPI Library** (e.g., OpenMPI or MPICH):
   - **Ubuntu**: `sudo apt install openmpi-bin openmpi-common libopenmpi-dev`
   - **macOS**: `brew install open-mpi`
4. **C++ Compiler** that supports C++11 or later.

---
