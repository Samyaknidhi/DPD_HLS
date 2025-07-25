	#include "dpd_joiner.h"

	// Compute |z|^2
	phi_t abs2(data_t i, data_t q) {
		return i*i + q*q;
	}

	// Iterative shifted Legendre polynomial (or similar orthogonal poly)
	phi_t iterative_P(int k, phi_t x) {
		phi_t P0 = 1, P1 = x - 1, Pk = 0;
		if (k == 0) return P0;
		if (k == 1) return P1;
		for (int n = 2; n <= k; n++) {
			Pk = ((2*n-1)*(x-1)*P1 - (n-1)*P0) * (1/n);
			P0 = P1;
			P1 = Pk;
		}
		return Pk;
	}

	// Compute all orthogonal polynomial basis functions for all memory taps
	void compute_phi_all(
		const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH],
		phi_t real_phi[K][MEMORY_DEPTH], phi_t imag_phi[K][MEMORY_DEPTH]
	) {
		for (int m = 0; m < MEMORY_DEPTH; ++m) {
			phi_t x = abs2(i_in[m], q_in[m]);
			for (int k = 0; k < K; ++k) {
	#pragma HLS UNROLL
				phi_t P = iterative_P(k, x);  // P_k(|z|^2)
				real_phi[k][m] = i_in[m] * P;
				imag_phi[k][m] = q_in[m] * P;
			}
		}
	}

	// Complex LMS DPD function (memory polynomial, orthogonal basis)
	void dpd(
		const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH],
		data_t i_ref, data_t q_ref,
		ccoef_t w[K][MEMORY_DEPTH],  // COMPLEX weights
		coef_t mu,
		data_t *i_out, data_t *q_out
	) {
		phi_t real_phi[K][MEMORY_DEPTH], imag_phi[K][MEMORY_DEPTH];
		compute_phi_all(i_in, q_in, real_phi, imag_phi);

		// Save previous weights
		ccoef_t w_prev[K][MEMORY_DEPTH];
		for (int k = 0; k < K; ++k)
			for (int m = 0; m < MEMORY_DEPTH; ++m)
				w_prev[k][m] = w[k][m];

		// Compute model output: complex multiply and accumulate
		acc_t sum_i = 0, sum_q = 0;
		for (int k = 0; k < K; ++k) {
			for (int m = 0; m < MEMORY_DEPTH; ++m) {
	#pragma HLS PIPELINE
				// (w_real + j*w_imag)*(phi_real + j*phi_imag) = (wr*pr - wi*pi) + j(wr*pi + wi*pr)
				sum_i += w_prev[k][m].real * real_phi[k][m] - w_prev[k][m].imag * imag_phi[k][m];
				sum_q += w_prev[k][m].real * imag_phi[k][m] + w_prev[k][m].imag * real_phi[k][m];
			}
		}
		*i_out = sum_i;
		*q_out = sum_q;

		// Compute error: e = x_ref - y_model
		data_t err_i = i_ref - *i_out;
		data_t err_q = q_ref - *q_out;

		// Complex LMS update
		for (int k = 0; k < K; ++k) {
			for (int m = 0; m < MEMORY_DEPTH; ++m) {
	#pragma HLS PIPELINE
				// w = w + mu * conj(phi) * err
				// (phi_real - j*phi_imag) * (err_i + j*err_q)
				// real: phi_real*err_i + phi_imag*err_q
				// imag: phi_real*err_q - phi_imag*err_i
		        update_t update_real = (update_t)err_i * real_phi[k][m] + (update_t)err_q * imag_phi[k][m];
		        update_t update_imag = (update_t)err_q * real_phi[k][m] - (update_t)err_i * imag_phi[k][m];
		        w[k][m].real = w_prev[k][m].real + mu * update_real;
		        w[k][m].imag = w_prev[k][m].imag + mu * update_imag;
			}
		}
	}
