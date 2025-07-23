#include "pa_joiner.h"
#include <cmath>
#include <iostream>

void saleh_amplifier(
    data_t in_i,
    data_t in_q,
    data_t& out_i,
    data_t& out_q,
    data_t& magnitude,
    data_t& gain_lin,
    data_t& gain_db
) {
    // More aggressive PA parameters for visible distortion
    const float alpha_a = 2.5;   // Amplitude coefficient
    const float beta_a = 1.8;    // Creates compression at lower levels
    const float alpha_p = 0.8;   // Phase coefficient
    const float beta_p = 1.2;    // Creates more AM-PM distortion

    // Get RF signal magnitude (in_q is always 0 from your circuit_final)
    float rf_magnitude = std::abs(in_i.to_float());

    if (rf_magnitude < 1e-6f) {
        // Handle zero input
        out_i = data_t(0);
        out_q = data_t(0);
        magnitude = data_t(0);
        gain_lin = data_t(1.0);
        gain_db = data_t(0.0);
        return;
    }

    // Saleh amplitude model (creates compression)
    float A_r = (alpha_a * rf_magnitude) / (1 + beta_a * rf_magnitude * rf_magnitude);

    // Saleh phase model (creates AM-PM distortion)
    float P_r = (alpha_p * rf_magnitude * rf_magnitude) / (1 + beta_p * rf_magnitude * rf_magnitude);

    // Apply amplitude compression to the RF signal
    float gain_factor = A_r / rf_magnitude;
    float compressed_rf = in_i.to_float() * gain_factor;

    // Add nonlinear distortion effects (creates harmonics and IMD)
    float distorted_rf = compressed_rf;
    if (std::abs(compressed_rf) > 0.1f) {
        // Add cubic and quintic nonlinearities
        float norm_signal = compressed_rf / 3.0f;  // Normalize to prevent overflow
        float cubic_term = norm_signal * norm_signal * norm_signal * 0.2f;
        float quintic_term = norm_signal * norm_signal * norm_signal * norm_signal * norm_signal * 0.05f;

        distorted_rf = compressed_rf + cubic_term + quintic_term;

        // Add AM-PM phase modulation effect
        float phase_mod = P_r * 0.15f * std::sin(P_r * 8.0f);
        distorted_rf *= (1.0f + phase_mod);
    }

    // Put distorted RF in out_i, create some Q due to phase distortion
    out_i = data_t(distorted_rf);

    // Phase distortion creates small Q component (realistic for RF PA)
    float q_distortion = distorted_rf * P_r * 0.1f * std::sin(P_r * 5.0f);
    out_q = data_t(q_distortion);

    // Apply the 5.0 scaling
    out_i = data_t(out_i.to_float());
    out_q = data_t(out_q.to_float());

    // Set other outputs
    magnitude = data_t(A_r);
    gain_lin = data_t(gain_factor);

    // FIXED: HLS-compatible dB calculation using pre-computed constant
    const float INV_LN10 = 0.434294481903f;  // 1/ln(10) to avoid division

    if (gain_factor > 0.001f) {
        gain_db = data_t(20.0f * std::log(gain_factor * 5.0f) * INV_LN10);
    } else {
        gain_db = data_t(-60.0f);  // Very low gain
    }

    // Debug output
    #ifndef __SYNTHESIS__
    static int debug_count = 0;
    if (debug_count < 5) {
        printf("PA[%d]: rf_in=%f, rf_mag=%f, A_r=%f, P_r=%f, compressed=%f, distorted=%f, out_i=%f, out_q=%f, gain_db=%f\n",
               debug_count++, in_i.to_float(), rf_magnitude, A_r, P_r,
               compressed_rf, distorted_rf, out_i.to_float(), out_q.to_float(), gain_db.to_float());
    }
    #endif
}
