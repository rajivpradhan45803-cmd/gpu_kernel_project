from experiments import run_experiments


def main():
    print("Running GPU Kernel Scientist (Mini Version)\n")

    results = run_experiments()

    # Select best configuration (minimum time)
    best = min(results, key=lambda x: x[2])

    print("\nBest Configuration:")
    print(f"{best[0]} with size {best[1]} -> {best[2]:.4f} seconds")


if __name__ == "__main__":
    main()