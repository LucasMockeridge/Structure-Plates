# Code to train and apply Pix2Pix
# Execute within https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# Create training data
python datasets/combine_A_and_B.py --fold_A ./datasets/obc/A --fold_B ./datasets/obc/B --fold_AB ./datasets/obc

# Train Pix2Pix
python train.py --dataroot ./datasets/obc --name obc_pix2pix --model pix2pix --input_nc 1 --output_nc 1 --no_flip --gpu_ids -1 --display_id 0

# Test Pix2Pix
python test.py --dataroot ./datasets/obc --name obc_pix2pix --model pix2pix --gpu_ids -1 --input_nc 1 --output_nc 1 --no_flip
