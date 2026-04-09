# gpu_kernel_project
This project is a simplified implementation inspired by the paper “GPU Kernel Scientist,” focusing on iterative performance optimization through experimentation and benchmarking. The objective is to understand how different computational configurations can be evaluated and improved based on execution results. As part of the task, I studied the provided resources, including the Hugging Face optimization guide and the research paper, to grasp the core concepts of optimization and experimentation.

I implemented a basic system using PyTorch that performs matrix multiplication under different configurations such as float32, float16, and bfloat16. The program measures execution time using high-precision timing and runs experiments across multiple matrix sizes. It then compares the results and identifies the best-performing configuration. Additionally, I incorporated CPU/GPU detection to make the implementation adaptable to available hardware.

This work represents an initial step (approximately 10–15%) toward building a system that follows an experimental and iterative approach to optimization.
