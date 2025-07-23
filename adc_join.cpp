#include <ap_int.h>
#include <ap_fixed.h>

#define N 8192
#define W 16  // ADC resolution bits

// FIXED: ADC that provides proper quantization AND compatible scaling
void dual_adc_system(
    ap_fixed<24,8> I_analog_in[N], ap_fixed<24,8> Q_analog_in[N],
    ap_fixed<24,8> I_digital_out[N], ap_fixed<24,8> Q_digital_out[N]
) {
#pragma HLS INTERFACE ap_memory port=I_analog_in
#pragma HLS INTERFACE ap_memory port=Q_analog_in
#pragma HLS INTERFACE ap_memory port=I_digital_out
#pragma HLS INTERFACE ap_memory port=Q_digital_out
#pragma HLS INTERFACE ap_ctrl_hs port=return

    const ap_fixed<24,8> V_REF = 1.0;
    const ap_fixed<24,8> V_MIN = -V_REF;
    const ap_fixed<24,8> V_MAX = V_REF;

    // ADC quantization parameters
    const int ADC_LEVELS = (1 << W);  // 65536 levels for 16-bit
    const ap_fixed<24,8> ADC_STEP = (V_MAX - V_MIN) / (ADC_LEVELS - 1);  // ~3.05e-5V per step

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1

        // I channel: True ADC quantization
        ap_fixed<24,8> clamped_I = (I_analog_in[i] > V_MAX) ? V_MAX :
                                   ((I_analog_in[i] < V_MIN) ? V_MIN : I_analog_in[i]);

        // Step 1: Convert to quantization level (0 to 65535)
        ap_fixed<24,8> normalized_I = (clamped_I - V_MIN) / ADC_STEP;
        int quantized_level_I = (int)(normalized_I + ap_fixed<24,8>(0.5));  // Round to nearest level

        // Step 2: Clamp to valid ADC range
        if (quantized_level_I < 0) quantized_level_I = 0;
        if (quantized_level_I >= ADC_LEVELS) quantized_level_I = 1 - ADC_LEVELS;

        // Step 3: Convert back to voltage (this is the "digital" representation)
        // This maintains signal levels while providing quantization effects
        I_digital_out[i] = V_MIN + (ap_fixed<24,8>(quantized_level_I) * ADC_STEP);

        // Q channel: Same process
        ap_fixed<24,8> clamped_Q = (Q_analog_in[i] > V_MAX) ? V_MAX :
                                   ((Q_analog_in[i] < V_MIN) ? V_MIN : Q_analog_in[i]);

        ap_fixed<24,8> normalized_Q = (clamped_Q - V_MIN) / ADC_STEP;
        int quantized_level_Q = (int)(normalized_Q + ap_fixed<24,8>(0.5));

        if (quantized_level_Q < 0) quantized_level_Q = 0;
        if (quantized_level_Q >= ADC_LEVELS) quantized_level_Q = ADC_LEVELS - 1;

        Q_digital_out[i] = V_MIN + (ap_fixed<24,8>(quantized_level_Q) * ADC_STEP);

        // Debug: Show quantization effect
        #ifndef __SYNTHESIS__
        if (i < 10 || i > N-10) {
            printf("Sample %d: I_analog=%f -> I_digital=%f (level=%d, diff=%f) | Q_analog=%f -> Q_digital=%f (level=%d, diff=%f)\n",
                   i, clamped_I.to_double(), I_digital_out[i].to_double(), quantized_level_I,
                   (I_digital_out[i] - clamped_I).to_double(),
                   clamped_Q.to_double(), Q_digital_out[i].to_double(), quantized_level_Q,
                   (Q_digital_out[i] - clamped_Q).to_double());
        }
        #endif
    }
}
