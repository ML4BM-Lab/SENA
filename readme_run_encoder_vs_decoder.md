
# -------------------------------------------- efficiency --------------------------------------------------
# sena encoder
nohup python3 -u encoder_vs_decoder.py senaenc_0 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_0_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_1 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_1_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_2 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_2_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_3 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_3_1layer_efficiency_norman_beta1.log &

# sena decoder
nohup python3 -u encoder_vs_decoder.py senadec_0 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_0_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_1 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_1_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_2 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_2_1layer_efficiency_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_3 efficiency norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_3_1layer_efficiency_norman_beta1.log &


# -------------------------------------------- interpretability --------------------------------------------------
# sena encoder
nohup python3 -u encoder_vs_decoder.py senaenc_0 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_0_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_1 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_1_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_2 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_2_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senaenc_3 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senaenc_3_1layer_interpretability_norman_beta1.log &

# sena decoder
nohup python3 -u encoder_vs_decoder.py senadec_0 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_0_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_1 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_1_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_2 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_2_1layer_interpretability_norman_beta1.log &

nohup python3 -u encoder_vs_decoder.py senadec_3 interpretability norman 1 1 > ./../../logs/encoder_vs_decoder/vae_senadec_3_1layer_interpretability_norman_beta1.log &
