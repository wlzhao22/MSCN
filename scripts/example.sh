# PSM
python main.py --anomaly_ratio 1 \
                --num_epochs 20 \
                --batch_size 256 \
                --mode train \
                --dataset PSM \
                --data_path ./dataset/PSM \
                --input_c 25 \
                --output_c 25 \
                --n_memory 25 \
                --lambd 0.00 \
                --lr 5e-4 \
                --memory_initial False \
                --phase_type None \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone MSCN \
                --metrics PA\

python main.py --anomaly_ratio 1.0 \
                --num_epochs 20  \
                --batch_size 512  \
                --mode memory_initial \
                --dataset PSM  \
                --data_path ./dataset/PSM  \
                --input_c 25 \
                --output_c 25 \
                --n_memory 25 \
                --lambd 0.1 \
                --lr 1e-4 \
                --memory_initial True \
                --phase_type second_train \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone MSCN \
                #--metrics PA \

test
python main.py --anomaly_ratio 1 \
                --num_epochs 10  \
                --batch_size 512  \
                --mode test \
                --dataset PSM  \
                --data_path ./dataset/PSM  \
                --lambd 0.1 \
                --input_c 25 \
                --output_c 25 \
                --n_memory 25 \
                --lr 1e-7 \
                --memory_initial False \
                --phase_type test \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone MSCN \
                # --metrics PA \
