python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_structure.yaml -d 1 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_dice_bce.yaml -d 1 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_tversky.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_tversky_bce.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_sufl.yaml -d 0