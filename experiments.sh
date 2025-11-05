#Baseline
echo "Running baseline with 0 trainable (dinov2) attention blocks"
python train.py --backbone dinov2_vitb14 --epochs 5 --warmup 2

#Baseline with norm layer
echo "Running baseline with 0 trainable (dinov2) attention blocks and norm layer"
python train.py --backbone dinov2_vitb14 --epochs 5 --warmup 2 --norm_layer

#Baseline, but DINOv3
echo "Running baseline with DINOv3 and 0 trainable attention blocks"
python train.py --backbone dinov3_vitb16 --epochs 5 --warmup 2

#DINOv3 backbone + cls token
echo "Running with backbone DINOv3 and appending cls token"
python train.py --backbone dinov3_vitb16 --epochs 5 --warmup 2 --cls_token

#DINOv3 backbone + norm layer. Best BoQ with DINOv3
echo "Running with backbone DINOv3 with norm layer enabled"
python train.py --backbone dinov3_vitb16 --epochs 5 --warmup 2 --norm_layer

#DINOv3 backbone + cls_token + norm layer
echo "Running with backbone DINOv3 with norm layer enabled and cls token"
python train.py --backbone dinov3_vitb16 --epochs 5 --warmup 2 --norm_layer --cls_token
