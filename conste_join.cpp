#include <ap_int.h>
#include <ap_fixed.h>

// CHANGED: Use ap_fixed<24,4> instead of ap_fixed<16,2>
typedef ap_fixed<24,8> fixed_point_t;

// Define maximum sizes
#define MAX_INPUT_BYTES 8192
#define MAX_SYMBOLS 32768

// Rest of your conste function remains the same...
void conste(
    ap_fixed<16,8> input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    fixed_point_t output_symbols_I[MAX_SYMBOLS],
    fixed_point_t output_symbols_Q[MAX_SYMBOLS]
) {
    // Your existing implementation - no changes needed
    #pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
    #pragma HLS INTERFACE s_axilite port=num_bits bundle=CRTL_BUS
    #pragma HLS INTERFACE s_axilite port=modulation_type bundle=CRTL_BUS
    #pragma HLS INTERFACE m_axi depth=MAX_INPUT_BYTES port=input_bytes offset=slave bundle=INPUT_BUS
    #pragma HLS INTERFACE m_axi depth=MAX_SYMBOLS port=output_symbols_I offset=slave bundle=OUTPUT_BUS_I
    #pragma HLS INTERFACE m_axi depth=MAX_SYMBOLS port=output_symbols_Q offset=slave bundle=OUTPUT_BUS_Q

    // QPSK constellation points (same values, just different type)
    const fixed_point_t qpsk_i[4] = {0.7071, -0.7071, 0.7071, -0.7071};
    const fixed_point_t qpsk_q[4] = {0.7071, 0.7071, -0.7071, -0.7071};

    int bits_per_symbol = 2; // QPSK uses 2 bits per symbol
    int max_symbols = num_bits / bits_per_symbol;

    // Process only up to the maximum number of symbols
    if (max_symbols > MAX_SYMBOLS) {
        max_symbols = MAX_SYMBOLS;
    }

    // Main processing loop
    process_symbols: for (int i = 0; i < max_symbols; i++) {
        #pragma HLS PIPELINE II=1

        // Extract 2 bits for QPSK
        int bit_pos = i * bits_per_symbol;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        // Bounds check to make HLS happy
        if (byte_idx >= MAX_INPUT_BYTES || byte_idx + 1 >= MAX_INPUT_BYTES) {
            // If out of bounds, set to zero
            output_symbols_I[i] = 0;
            output_symbols_Q[i] = 0;
            continue;
        }

        // Extract the symbol bits
        ap_uint<2> symbol_bits;

        if (bit_offset <= 6) {
            // Bits are in the same byte
            symbol_bits = (input_bytes[byte_idx] >> (6 - bit_offset)) & 0x3;
        } else {
            // Bits are split across two bytes
            symbol_bits = ((input_bytes[byte_idx] & 0x1) << 1) |
                          ((input_bytes[byte_idx + 1] >> 7) & 0x1);
        }

        // Map to constellation points
        output_symbols_I[i] = qpsk_i[symbol_bits];
        output_symbols_Q[i] = qpsk_q[symbol_bits];
    }
}
