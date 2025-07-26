import numpy as np
import matplotlib.pyplot as plt

def generate_16qam_symbols(num_symbols=8192, filename_i="i_symbols.txt", filename_q="q_symbols.txt"):
    """
    Generates a random 16-QAM signal and saves the I and Q components to text files.

    A 16-QAM signal is a good test signal for communication systems because it is
    "noise-like" and not a simple repeating pattern. This allows for accurate
    cross-correlation measurements to find system delays.

    Args:
        num_symbols (int): The number of QAM symbols to generate. This should
                           match the DATA_LEN in your C++ project.
        filename_i (str): The output filename for the I-component symbols.
        filename_q (str): The output filename for the Q-component symbols.
    """
    print(f"Generating {num_symbols} random 16-QAM symbols...")

    # 1. Define the 16-QAM constellation points
    # The points are typically at {-3, -1, 1, 3} for both I and Q axes.
    # We scale them by 1/sqrt(10) to normalize the average power to 1.0.
    qam_points = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    
    # 2. Generate random integers to select points from the constellation
    # We need 'num_symbols' random choices for the I-axis and 'num_symbols' for the Q-axis.
    i_indices = np.random.randint(0, 4, num_symbols)
    q_indices = np.random.randint(0, 4, num_symbols)
    
    # 3. Create the I and Q symbol streams by mapping the random indices
    # to the actual constellation point values.
    i_symbols = qam_points[i_indices]
    q_symbols = qam_points[q_indices]

    # 4. Save the generated I and Q symbols to text files
    # The format is one floating-point number per line, with high precision,
    # which is what your C++ testbench expects.
    try:
        # Using '%.16f' to format the numbers with 16 decimal places as requested.
        np.savetxt(filename_i, i_symbols, fmt='%.16f')
        print(f"Successfully saved I symbols to '{filename_i}'")
        
        np.savetxt(filename_q, q_symbols, fmt='%.16f')
        print(f"Successfully saved Q symbols to '{filename_q}'")

    except Exception as e:
        print(f"Error: Could not write to files. {e}")
        return

    # 5. Plot the results for visual confirmation
    plt.figure(figsize=(12, 6))

    # Plot the first 100 symbols to show the noise-like nature
    plt.subplot(2, 1, 1)
    plt.title(f'Generated 16-QAM Signal (First 100 Symbols)')
    plt.plot(i_symbols[:100], 'o-', label='I Component')
    plt.plot(q_symbols[:100], 'o-', label='Q Component')
    plt.xlabel('Symbol Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot the constellation diagram to confirm it's 16-QAM
    plt.subplot(2, 1, 2)
    plt.title('Constellation Diagram')
    plt.scatter(i_symbols, q_symbols, marker='o', color='blue')
    plt.xlabel('I Component (In-Phase)')
    plt.ylabel('Q Component (Quadrature)')
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid(True)
    plt.axis('equal') # Ensure I and Q axes have the same scale

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate 8192 symbols to match DATA_LEN in the C++ project.
    generate_16qam_symbols(num_symbols=8192)
