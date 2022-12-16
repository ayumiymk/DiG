from timm.models.registry import register_model
from timm.models import create_model

import modeling_finetune

def create_encoder(args):
  encoder = create_model(
      args.model,
      pretrained=False,
      # num_classes=args.nb_classes,
      num_classes=0,
      drop_rate=args.drop,
      drop_path_rate=args.drop_path,
      attn_drop_rate=args.attn_drop_rate,
      drop_block_rate=None,
      use_mean_pooling=args.use_mean_pooling,
      init_scale=args.init_scale,
      return_feat_map=not args.use_seq_cls_token,
  )
  return encoder