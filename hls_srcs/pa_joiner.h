#ifndef AMPLIFIER_HERE_H
#define AMPLIFIER_HERE_H

#include "dpd_joiner.h"

// Saleh amplifier model
void saleh_amplifier(
    data_t in_i,
    data_t in_q,
    data_t& out_i,
    data_t& out_q,
    data_t& magnitude,
    data_t& gain_lin,
    data_t& gain_db
);

#endif // AMPLIFIER_HERE_H
