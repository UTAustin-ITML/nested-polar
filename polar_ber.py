#!/usr/bin/env python3

import sys
sys.path.insert(0, '../py_aff3ct/build/lib')

import numpy as np
import py_aff3ct as aff3ct
import math
import time
import matplotlib.pyplot as plt
from datetime import timedelta

def polar_ber(N, K_crc, channel, min_err, frozen_bits, SNRs, decoder_type = "sc_naive", list_size = 1, crc_len = 0, n_thread = 2):
    
	if crc_len == 4:
		crc_poly = '4-ITU'
	elif crc_len == 11:
		crc_poly = '11-5G'
	else:
		crc_poly = None
  
	K = K_crc - crc_len
	
	ebn0_min = SNRs[0]
	ebn0_step = 0.25
	ebn0_max = SNRs[-1] + ebn0_step

	ebn0 = np.arange(ebn0_min,ebn0_max,ebn0_step)
	esn0 = ebn0 #+ 10 * math.log10(K/N)
	sigma_vals = 1/(math.sqrt(2) * 10 ** (esn0 / 20))
   
	src = aff3ct.module.source.Source_random_fast(K, 12)
 
	mdm = aff3ct.module.modem.Modem_BPSK_fast(N)
	gen = aff3ct.tools.Gaussian_noise_generator_implem.FAST
	if channel == "awgn":
		chn = aff3ct.module.channel.Channel_AWGN_LLR(N, gen)
	elif channel == "rayleigh":
		chn = aff3ct.module.channel.Channel_Rayleigh_LLR(N, False, gen)
	mnt = aff3ct.module.monitor.Monitor_BFER_AR(K, min_err)
 
	sigma = np.ndarray(shape = (1,1),  dtype = np.float32)

	if decoder_type == "scl_crc":

		crc = aff3ct.module.crc.CRC_polynomial_fast(K,crc_poly)
		enc = aff3ct.module.encoder.Encoder_polar_sys(K_crc,N,frozen_bits)
		dec = aff3ct.module.decoder.Decoder_polar_SCL_fast_CA_sys(K_crc,N,list_size,frozen_bits,crc) # Use CRC for the SCL CA

		crc["build        ::U_K1"].bind(src["generate   ::U_K "])
		enc["encode       ::U_K "].bind(crc["build      ::U_K2"])

		mdm["modulate     ::X_N1"].bind(enc["encode     ::X_N "])
  
		if channel == "awgn":
			chn["add_noise    ::X_N "].bind(mdm["modulate   ::X_N2"])
			mdm["demodulate   ::Y_N1"].bind(chn["add_noise  ::Y_N "])
		elif channel == "rayleigh":
			chn["add_noise_wg    ::X_N "].bind(mdm["modulate   ::X_N2"])
			mdm["demodulate   ::Y_N1"].bind(chn["add_noise_wg  ::Y_N "])
  
		dec["decode_siho  ::Y_N "].bind(mdm["demodulate ::Y_N2"])
	
		crc["extract      ::V_K1"].bind(dec["decode_siho::V_K "])
		mnt["check_errors ::U   "].bind(src["generate   ::U_K "])
		mnt["check_errors ::V   "].bind(crc["extract    ::V_K2"])
	else:
	 
		if decoder_type == "sc_naive":
			enc = aff3ct.module.encoder.Encoder_polar(K,N,frozen_bits)
			dec = aff3ct.module.decoder.Decoder_polar_SC_naive(K,N,frozen_bits)
		elif decoder_type == "sc_sys":
			enc = aff3ct.module.encoder.Encoder_polar_sys(K,N,frozen_bits)
			dec = aff3ct.module.decoder.Decoder_polar_SC_fast_sys(K,N,frozen_bits)
		elif decoder_type == "scl":
			enc = aff3ct.module.encoder.Encoder_polar(K,N,frozen_bits)
			dec = aff3ct.module.decoder.Decoder_polar_SCL_naive(K,N,list_size,frozen_bits)

		enc["encode       ::U_K "].bind(src["generate   ::U_K "])
  
		mdm["modulate     ::X_N1"].bind(enc["encode     ::X_N "])
  
		if channel == "awgn":
			chn["add_noise    ::X_N "].bind(mdm["modulate   ::X_N2"])
			mdm["demodulate   ::Y_N1"].bind(chn["add_noise  ::Y_N "])
		elif channel == "rayleigh":
			chn["add_noise_wg    ::X_N "].bind(mdm["modulate   ::X_N2"])
			mdm["demodulate   ::Y_N1"].bind(chn["add_noise_wg  ::Y_N "])
  
		dec["decode_siho  ::Y_N "].bind(mdm["demodulate ::Y_N2"])
  
		mnt["check_errors ::U   "].bind(src["generate   ::U_K "])
		mnt["check_errors ::V   "].bind(dec["decode_siho::V_K "])
  
	if channel == "awgn":
		chn["add_noise    ::CP  "].bind(                 sigma  )
	elif channel == "rayleigh":
		chn["add_noise_wg    ::CP  "].bind(                 sigma  )
	mdm["demodulate   ::CP  "].bind(                 sigma  )

	#src("generate"    ).stats = True
	#enc("encode"      ).stats = True
	#mdm("modulate"    ).stats = True
	#chn("add_noise"   ).stats = True
	#mdm("demodulate"  ).stats = True
	#dec("decode_siho" ).stats = True
	#mnt("check_errors").stats = True

	seq  = aff3ct.tools.sequence.Sequence(src["generate"], mnt["check_errors"], n_thread)

	fer = np.zeros(len(ebn0))
	ber = np.zeros(len(ebn0))
 
	sigma[:] = sigma_vals[0]
 
	enc.set_frozen_bits(frozen_bits)
	dec.set_frozen_bits(frozen_bits)
 
	seq.exec()

	ber = mnt.get_ber()
	fer = mnt.get_fer()

	mnt.reset()

	return ber, fer
