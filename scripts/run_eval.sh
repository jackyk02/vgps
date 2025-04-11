#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

model_name=octo-small

# WidowX tasks: widowx_put_eggplant_in_basket, widowx_spoon_on_towel, widowx_carrot_on_plate, widowx_stack_cube
# Google Robot tasks: google_robot_pick_coke_can, google_robot_move_near

# Start the FastAPI server in the background
python scripts/server_vgps.py --checkpoint="/root/V-GPS/v-gps" --port=8000