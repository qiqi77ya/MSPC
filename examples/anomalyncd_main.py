# 加MS2

import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='AnomalyNCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ----------------------
    # dataset setting
    # ----------------------
    parser.add_argument('--dataset', type=str, default='mvtec', help='novel dataset name')
    parser.add_argument('--category', default=None, type=str, help='novel category name')
    parser.add_argument('--dataset_path', type=str, default=None, help='input novel image path')
    parser.add_argument('--anomaly_map_path', type=str, default=None, help='input novel anomaly map path')
    parser.add_argument('--binary_data_path', type=str, default=None, help='output novel binary mask path')
    parser.add_argument('--crop_data_path', type=str, default=None, help='output novel crop data path')
    parser.add_argument('--base_data_path', default=None, type=str, help='input base image path')

    # ----------------------
    # experiment setting
    # ----------------------
    parser.add_argument('--config', type=str, default='./configs/AnomalyNCD.yaml', help='config file path')
    parser.add_argument('--runner_name', default='AnomalyNCD', type=str)
    parser.add_argument('--only_test', type=str, default=None, help='test using the trained checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path of the trained checkpoint')

    args = parser.parse_args()

    return args


def load_args(cfg, args):
    """
    Load args from the config file
    """
    # ----------------------
    # binariation setting
    # ----------------------
    args.sample_rate = cfg['binarization']['sample_rate']
    args.min_interval_len = cfg['binarization']['min_interval_len']
    args.erode = cfg['binarization']['erode']
    # ----------------------
    # model setting
    # ----------------------
    args.grad_from_block = cfg['models']['grad_from_block']
    args.pretrained_backbone = cfg['models']['pretrained_backbone']
    args.mask_layers = cfg['models']['mask_layers']
    args.ms_layers = cfg['models'].get('ms_layers', [3, 6, 9, 12])
    args.feature_fusion = cfg['models'].get('feature_fusion', 'cross_attention')
    args.aux_layers = cfg['models'].get('aux_layers', [6, 9])
    args.n_views = cfg['models']['n_views']
    args.n_head = cfg['models']['n_head']
    # ----------------------
    # training setting
    # ----------------------
    args.batch_size = cfg['training']['batch_size']
    args.num_workers = cfg['training']['num_workers']
    args.lr = cfg['training']['lr']
    args.gamma = cfg['training']['gamma']
    args.momentum = cfg['training']['momentum']
    args.weight_decay = cfg['training']['weight_decay']
    args.epochs = cfg['training']['epochs']
    # ----------------------
    # loss setting
    # ----------------------
    args.sup_weight = cfg['loss']['sup_weight']
    args.memax_weight = cfg['loss']['memax_weight']
    args.anomaly_thred = cfg['loss']['anomaly_thred']
    args.teacher_temp = cfg['loss']['teacher_temp']
    args.warmup_teacher_temp = cfg['loss']['warmup_teacher_temp']
    args.warmup_teacher_temp_epochs = cfg['loss']['warmup_teacher_temp_epochs']
    args.repeat_times = cfg['loss']['repeat_times']
    args.consistency_weight = cfg['loss'].get('consistency_weight', 0.3)
    args.consistency_distill_temp = cfg['loss'].get('consistency_distill_temp', 2.0)
    args.consistency_gate_temp = cfg['loss'].get('consistency_gate_temp', 1.0)
    args.consistency_scope = cfg['loss'].get('consistency_scope', 'unlabeled_high_conf')
    args.consistency_select_mode = cfg['loss'].get('consistency_select_mode', 'top_ratio')
    args.consistency_conf_thresh = cfg['loss'].get('consistency_conf_thresh', 0.2)
    args.consistency_top_ratio = cfg['loss'].get('consistency_top_ratio', 0.5)
    args.consistency_warmup_epochs = cfg['loss'].get('consistency_warmup_epochs', 5)
    args.consistency_fallback_to_unlabeled = cfg['loss'].get('consistency_fallback_to_unlabeled', True)
    args.proto_weight          = cfg['loss'].get('proto_weight', 0.1)
    args.proto_momentum        = cfg['loss'].get('proto_momentum', 0.95)
    args.proto_temperature     = cfg['loss'].get('proto_temperature', 0.07)
    args.proto_conf_threshold  = cfg['loss'].get('proto_conf_threshold', 0.5)
    inference_cfg = cfg.get('inference', {})
    args.use_confidence_area_merge = inference_cfg.get('use_confidence_area_merge', True)
    args.confidence_area_alpha = inference_cfg.get('confidence_area_alpha', 1.0)
    args.confidence_area_beta = inference_cfg.get('confidence_area_beta', 1.0)
    args.confidence_area_eps = inference_cfg.get('confidence_area_eps', 1e-6)
    args.write_carm_metrics = inference_cfg.get('write_carm_metrics', True)
    args.carm_conf_temp = inference_cfg.get('carm_conf_temp', 1.0)
    args.carm_area_gamma = inference_cfg.get('carm_area_gamma', 0.5)
    args.carm_use_density = inference_cfg.get('carm_use_density', True)
    args.carm_normalize_signals = inference_cfg.get('carm_normalize_signals', True)
    args.carm_use_head_agreement = inference_cfg.get('carm_use_head_agreement', True)
    args.carm_agreement_weight = inference_cfg.get('carm_agreement_weight', 0.5)
    args.carm_use_margin_signal = inference_cfg.get('carm_use_margin_signal', True)
    args.carm_margin_mode = inference_cfg.get('carm_margin_mode', 'top2_entropy')
    args.carm_margin_weight = inference_cfg.get('carm_margin_weight', 0.35)
    args.carm_use_head_kl = inference_cfg.get('carm_use_head_kl', True)
    args.carm_head_kl_weight = inference_cfg.get('carm_head_kl_weight', 0.25)
    # ──────────────────────────────────────────────────────────────────
    # ----------------------
    # experiment setting
    # ----------------------
    args.seed = cfg['experiment']['seed']
    args.print_freq = cfg['experiment']['print_freq']
    args.table_root = cfg['experiment']['table_root']
    args.exp_name = cfg['experiment']['exp_name']
    args.exp_root = cfg['experiment']['exp_root']

    return args


if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    args = load_args(cfg, args)
    model = AnomalyNCD(args)
    model.main()
