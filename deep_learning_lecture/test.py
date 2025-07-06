import numpy as np
import matplotlib.pyplot as plt


def simulate_deep_network_gradients():
    """Simulate gradient flow in deep networks"""

    num_layers = 15  # Deeper network
    num_trials = 1000

    sigmoid_final_gradients = []
    relu_final_gradients = []

    for trial in range(num_trials):
        # Sigmoid: Each layer multiplies gradient by 0.1-0.25
        sigmoid_layer_grads = np.random.uniform(0.1, 0.25, num_layers)
        sigmoid_final = np.prod(sigmoid_layer_grads)
        sigmoid_final_gradients.append(sigmoid_final)

        # ReLU: Each layer either kills (0) or preserves (1) gradient
        # 70% chance of survival per layer
        relu_layer_grads = np.random.choice([0, 1], num_layers, p=[0.3, 0.7])
        relu_final = np.prod(relu_layer_grads)
        relu_final_gradients.append(relu_final)

    return sigmoid_final_gradients, relu_final_gradients


def simulate_learning_with_gradients():
    """Simulate actual learning process with gradient magnitudes"""

    epochs = 100
    learning_rate = 0.1

    # Initial loss
    sigmoid_loss = 2.0
    relu_loss = 2.0

    sigmoid_losses = [sigmoid_loss]
    relu_losses = [relu_loss]

    for epoch in range(epochs):
        # Simulate gradient calculation for 15-layer network

        # Sigmoid gradient (vanishing)
        sigmoid_grad = np.random.uniform(1e-8, 1e-6)  # Very small gradients
        sigmoid_loss -= learning_rate * sigmoid_grad
        sigmoid_loss = max(sigmoid_loss, 0.1)  # Prevent negative loss
        sigmoid_losses.append(sigmoid_loss)

        # ReLU gradient (either 0 or normal size)
        if np.random.random() < 0.15:  # 15% chance of effective gradient
            relu_grad = np.random.uniform(0.01, 0.1)  # Normal gradient
            relu_loss -= learning_rate * relu_grad
        # else: no gradient (dying ReLU)
        relu_loss = max(relu_loss, 0.1)
        relu_losses.append(relu_loss)

    return sigmoid_losses, relu_losses


def compare_activation_functions():
    """Compare different aspects of activation functions"""

    # 1. Gradient flow simulation
    sig_grads, relu_grads = simulate_deep_network_gradients()

    # 2. Learning simulation
    sig_losses, relu_losses = simulate_learning_with_gradients()

    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Final gradient distribution
    axes[0, 0].hist(sig_grads, bins=50, alpha=0.7, color="red", label="Sigmoid")
    axes[0, 0].set_xlabel("Final Gradient Magnitude")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Sigmoid: Vanishing Gradients")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    axes[0, 1].hist(
        relu_grads, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, color="blue", label="ReLU"
    )
    axes[0, 1].set_xlabel("Final Gradient Magnitude")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("ReLU: Binary Gradients")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    # 2. Learning curves
    epochs = range(len(sig_losses))
    axes[0, 2].plot(epochs, sig_losses, "r-", label="Sigmoid", linewidth=2)
    axes[0, 2].plot(epochs, relu_losses, "b-", label="ReLU", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Learning Curves")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 3. Statistical comparison
    sig_mean = np.mean(sig_grads)
    relu_mean = np.mean(relu_grads)
    relu_survival = np.mean(np.array(relu_grads) > 0)

    axes[1, 0].bar(
        ["Sigmoid\nMean Grad", "ReLU\nMean Grad"],
        [sig_mean, relu_mean],
        color=["red", "blue"],
        alpha=0.7,
    )
    axes[1, 0].set_ylabel("Mean Gradient")
    axes[1, 0].set_title("Average Gradient Comparison")
    axes[1, 0].set_yscale("log")

    # 4. Survival rates
    axes[1, 1].bar(
        ["Sigmoid\nSurvival", "ReLU\nSurvival"],
        [1.0, relu_survival],
        color=["red", "blue"],
        alpha=0.7,
    )
    axes[1, 1].set_ylabel("Gradient Survival Rate")
    axes[1, 1].set_title("Path Survival Comparison")
    axes[1, 1].set_ylim(0, 1.2)

    # Add percentage labels
    axes[1, 1].text(0, 1.05, "100%", ha="center", fontweight="bold")
    axes[1, 1].text(
        1, relu_survival + 0.05, f"{relu_survival:.1%}", ha="center", fontweight="bold"
    )

    # 5. Final performance
    final_sig_loss = sig_losses[-1]
    final_relu_loss = relu_losses[-1]

    axes[1, 2].bar(
        ["Sigmoid\nFinal Loss", "ReLU\nFinal Loss"],
        [final_sig_loss, final_relu_loss],
        color=["red", "blue"],
        alpha=0.7,
    )
    axes[1, 2].set_ylabel("Final Loss (lower is better)")
    axes[1, 2].set_title("Final Learning Performance")

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("=== Deep Network Gradient Analysis ===")
    print(f"Sigmoid mean gradient: {sig_mean:.2e}")
    print(f"ReLU mean gradient: {relu_mean:.2e}")
    print(f"ReLU advantage: {relu_mean/sig_mean:.0f}x larger gradients")
    print(f"ReLU path survival rate: {relu_survival:.1%}")

    print(f"\nLearning Performance:")
    print(f"Sigmoid final loss: {final_sig_loss:.3f}")
    print(f"ReLU final loss: {final_relu_loss:.3f}")
    improvement = (final_sig_loss - final_relu_loss) / final_sig_loss * 100
    print(f"ReLU improvement: {improvement:.1f}%")


def demonstrate_layer_depth_effect():
    """Show how network depth affects gradient flow"""

    layer_depths = [3, 5, 10, 15, 20]
    sigmoid_final_grads = []
    relu_survival_rates = []

    for depth in layer_depths:
        # Sigmoid: multiply gradients
        sig_grad = 0.2**depth  # Each layer reduces gradient by 0.2
        sigmoid_final_grads.append(sig_grad)

        # ReLU: survival probability
        survival_prob = 0.7**depth  # 70% survival per layer
        relu_survival_rates.append(survival_prob)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(layer_depths, sigmoid_final_grads, "ro-", linewidth=2, markersize=8)
    plt.xlabel("Network Depth (layers)")
    plt.ylabel("Final Gradient Magnitude")
    plt.title("Sigmoid: Exponential Decay")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(layer_depths, relu_survival_rates, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Network Depth (layers)")
    plt.ylabel("Path Survival Probability")
    plt.title("ReLU: Survival Rate vs Depth")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("\n=== Effect of Network Depth ===")
    for i, depth in enumerate(layer_depths):
        print(
            f"Depth {depth:2d}: Sigmoid grad={sigmoid_final_grads[i]:.2e}, "
            f"ReLU survival={relu_survival_rates[i]:.3f}"
        )


def explain_why_original_sim_failed():
    """Explain why the original simulation didn't show the difference"""

    print("\n=== Why Original Simulation Showed No Difference ===")
    print(
        "\n1. Network was too shallow (only simulated final learning, not gradient flow)"
    )
    print("2. Learning rates were too conservative")
    print("3. Didn't simulate the actual backpropagation process")
    print("4. ReLU advantage appears mainly in DEEP networks (10+ layers)")

    print("\nKey Insight:")
    print("ReLU's advantage is NOT in the final learning step,")
    print("but in PRESERVING gradients through many layers during backpropagation!")

    print("\nAnalogy:")
    print(
        "- Sigmoid: Like whispering a message through 15 people - gets weaker each time"
    )
    print("- ReLU: Like passing a note - either passes perfectly or stops completely")
    print("  In deep networks, perfect occasional passing > consistent weakening")


# Run all demonstrations
if __name__ == "__main__":
    compare_activation_functions()
    demonstrate_layer_depth_effect()
    explain_why_original_sim_failed()
