wget -P ./weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ./GroundingDINO/weights/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -P ./GroundingDINO/weights/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
gdown -O ./OVSeg/weights/ https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing 
