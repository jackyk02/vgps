export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

model_name=octo-small

# WidowX tasks: widowx_put_eggplant_in_basket, widowx_spoon_on_towel, widowx_carrot_on_plate, widowx_stack_cube
# Google Robot tasks: google_robot_pick_coke_can, google_robot_move_near

for task_name in widowx_put_eggplant_in_basket
do
xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python experiments/eval_vgps.py \
--seed=0 \
--model_name=$model_name \
--task_name=$task_name \
--use_vgps=True \
--vgps_checkpoint="/root/V-GPS/v-gps" \
--num_samples=50 \
--action_temp=1.0 \
--num_eval_episodes=100 
done
