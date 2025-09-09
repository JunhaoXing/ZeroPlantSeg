# How to build

1. Install requirements.txt
```
$ python3 -m pip install -r requirements.txt
```

2. Install CLIP
```
$ cd OVSeg/third_party/CLIP
$ python3 -m pip install -Ue .
```
3. Install GroundingDINO
```
$ cd GroundingDINO
$ python3 -m pip install -e .
```

## in the case Error: "AttributeError: module 'PIL.Image' has no attribute 'LINEAR'"
```
$ python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```