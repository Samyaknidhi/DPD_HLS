#ifndef ADC_HERE_F_H
#define ADC_HERE_F_H

#include <ap_int.h>
#include <ap_fixed.h>

#define N 8192
#define W 16  // Output bit width

void dual_adc_system(
    ap_fixed<32,16> I_analog_in[N],
    ap_fixed<32,16> Q_analog_in[N],
    ap_fixed<32,16> I_digital_out[N],
    ap_fixed<32,16> Q_digital_out[N]
);

#endif // ADC_HERE_F_H
