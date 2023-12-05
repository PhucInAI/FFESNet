# ========================================================================
# CVC-ClinicDB
# ========================================================================
python train_LA.py -c configs/Loss_CVC-ClinicDB/LA_FFESNet_A2FPN_B0_drop0.5_CVC-ClinicDB_structure.yaml -d 0 && \
python train_LA.py -c configs/Loss_CVC-ClinicDB/LA_FFESNet_A2FPN_B0_drop0.5_CVC-ClinicDB_dice_bce.yaml -d 0 && \
python train_LA.py -c configs/Loss_CVC-ClinicDB/LA_FFESNet_A2FPN_B0_drop0.5_CVC-ClinicDB_tversky.yaml -d 0 && \
python train_LA.py -c configs/Loss_CVC-ClinicDB/LA_FFESNet_A2FPN_B0_drop0.5_CVC-ClinicDB_tversky_bce.yaml -d 0 && \
python train_LA.py -c configs/Loss_CVC-ClinicDB/LA_FFESNet_A2FPN_B0_drop0.5_CVC-ClinicDB_sufl.yaml -d 0

# ========================================================================
# Kvasir
# ========================================================================
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_structure.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_dice_bce.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_tversky.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_tversky_bce.yaml -d 0 && \
python train_LA.py -c configs/Loss_Kvasir/LA_FFESNet_A2FPN_B0_drop0.5_Kvasir_sufl.yaml -d 0