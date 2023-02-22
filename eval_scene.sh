rm -r tmp-eval-result
python predict.py --config $1 --input_dir datasets/evaluation/Tamper-Scene/i_s/ --save_dir tmp-eval-result --checkpoint $2 --slm
python eval_real.py --saved_model models/TPS-ResNet-BiLSTM-Attn.pth --gt_file datasets/evaluation/Tamper-Scene/i_t.txt --image_folder tmp-eval-result 
rm -r tmp-eval-result
