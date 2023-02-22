rm -r tmp-eval-result
python predict.py --config $1 --input_dir datasets/evaluation/Tamper-Syn2k/i_s/ --save_dir tmp-eval-result --checkpoint $2 --slm
python evaluation.py --gt_path datasets/evaluation/Tamper-Syn2k/t_f/ --target_path tmp-eval-result/
rm -r tmp-eval-result
