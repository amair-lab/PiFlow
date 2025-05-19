from AgenX_Nanohelix.core.models import predict_g_factor

import numpy as np


if __name__ == "__main__":

    # G-factor = 0.906427179
    result = predict_g_factor({
        'pitch': 160.0,
        'fiber_radius': 23.0,
        'n_turns': 4.0,
        'helix_radius': 50.0
    })

    # Display results
    print("\nNanohelix Structure Properties:")
    print("---------------------------------")
    print(f"Basic Parameters:")
    print(f"  Pitch: {result['pitch']:.2f}")
    print(f"  Fiber Radius: {result['fiber_radius']:.2f}")
    print(f"  Number of Turns: {result['n_turns']:.2f}")
    print(f"  Helix Radius: {result['helix_radius']:.2f}")

    print("\nDerived Parameters:")
    print(f"  Total Length: {result['total_length']:.2f}")
    print(f"  Curl: {result['curl']:.6f}")
    print(f"  Angle: {result['angle']:.6f} rad ({np.degrees(result['angle']):.2f}°)")
    print(f"  Height: {result['height']:.2f}")
    print(f"  Total Fiber Length: {result['total_fiber_length']:.2f}")
    print(f"  Volume: {result['V']:.2f}")
    print(f"  Mass: {result['mass']:.2f}")

    print(f"\nPredicted G-Factor: {result['g_factor']:.6f}")