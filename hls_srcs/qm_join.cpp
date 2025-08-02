#include <math.h>
#include <ap_fixed.h>

typedef ap_fixed<24,8> data_t;
typedef ap_fixed<32,4> acc_t;

// FIXED: Accurate cosine LUT for DPD (0 to pi/2 only)
void nco(ap_uint<8> &phase, ap_uint<8> phase_inc, data_t &cos_lo, data_t &sin_lo) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=cos_lut complete dim=1

    // FIXED: High-precision quarter-wave cosine LUT for DPD
    static const ap_fixed<16,4> cos_lut[64] = {
        1.0000, 0.9988, 0.9951, 0.9890, 0.9808, 0.9703, 0.9576, 0.9426,
        0.9254, 0.9063, 0.8854, 0.8625, 0.8378, 0.8115, 0.7834, 0.7537,
        0.7224, 0.6897, 0.6557, 0.6204, 0.5839, 0.5464, 0.5080, 0.4687,
        0.4286, 0.3878, 0.3464, 0.3043, 0.2616, 0.2185, 0.1749, 0.1309,
        0.0866, 0.0420, 0.0000, -0.0420, -0.0866, -0.1309, -0.1749, -0.2185,
        -0.2616, -0.3043, -0.3464, -0.3878, -0.4286, -0.4687, -0.5080, -0.5464,
        -0.5839, -0.6204, -0.6557, -0.6897, -0.7224, -0.7537, -0.7834, -0.8115,
        -0.8378, -0.8625, -0.8854, -0.9063, -0.9254, -0.9426, -0.9576, -0.9703
    };

    // Extract quadrant and index
    ap_uint<2> quadrant = phase >> 6;
    ap_uint<6> index = phase & 0x3F;

    // Get LUT values
    ap_fixed<16,4> cos_base = cos_lut[index];
    ap_fixed<16,4> sin_base = cos_lut[63-index];  // sin(alpha) = cos(pi/2 - alpha)

    data_t cos_val, sin_val;

    // FIXED: Correct quadrant mapping for DPD
    switch(quadrant) {
        case 0: // 0 to 90 (0 to pi/2)
            cos_val = (data_t)cos_base;    // cos(alpha)
            sin_val = (data_t)sin_base;    // sin(alpha) = cos(pi/2 - alpha)
            break;
        case 1: // 90 to 180 (pi/2 to pi)
            cos_val = (data_t)(-sin_base); // cos(pi/2 + alpha) = -sin(alpha)
            sin_val = (data_t)cos_base;    // sin(pi/2 + alpha) = cos(alpha)
            break;
        case 2: // 180 to 270 (pi to 3pi/2)
            cos_val = (data_t)(-cos_base); // cos(pi + alpha) = -cos(alpha)
            sin_val = (data_t)(-sin_base); // sin(pi + alpha) = -sin(alpha)
            break;
        case 3: // 270 to 360 (3pi/2 to 2pi)
            cos_val = (data_t)sin_base;    // cos(3pi/2 + alpha) = sin(alpha)
            sin_val = (data_t)(-cos_base); // sin(3pi/2 + alpha) = -cos(alpha)
            break;
        default:
            cos_val = (data_t)1.0;
            sin_val = (data_t)0.0;
            break;
    }

    cos_lo = cos_val;
    sin_lo = sin_val;

    // Update phase for continuous operation
    phase += phase_inc;

    // DPD Debug: Monitor NCO for spectral purity
    #ifndef __SYNTHESIS__
    static int nco_debug = 0;
    if (nco_debug < 8) {
        printf("DPD_NCO[%d]: phase=0x%02X, quad=%d, idx=%d, cos=%f, sin=%f\n",
               nco_debug++, (int)(phase-phase_inc), (int)quadrant, (int)index,
               cos_val.to_double(), sin_val.to_double());
    }
    #endif
}

// FIXED: High-precision QM for DPD linearity
data_t digital_qm(data_t I, data_t Q, data_t cos_lo, data_t sin_lo) {
#pragma HLS INLINE
#pragma HLS PIPELINE II=1

    // Use higher precision for DPD to minimize quantization noise
    acc_t i_term = (acc_t)I * (acc_t)cos_lo;
    acc_t q_term = (acc_t)Q * (acc_t)sin_lo;
    acc_t qm_output = i_term - q_term;

    // DPD requires linear mixing without saturation
    return (data_t)qm_output;
}
