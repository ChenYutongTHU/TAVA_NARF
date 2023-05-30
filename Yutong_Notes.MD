# High-level structure

https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/launch.py#L64-L65

* engine: trainer/evaluator

## Trainer
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L33-L34
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L97-L98

### Abstract
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L13-L14
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L35
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L57
* build_model(), build_dataset() are defined by Trainer()

### DataLoader
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L77

#### [ZJU Config](configs/dataset/zju.yaml)
https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/configs/dataset/zju.yaml#L1
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/zju_loader.py#L46
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/abstract.py#L7
* abstract.py/CachedIterDataset also define __getitem__()
* fetch_data() and preprocess() are defined by SubjectLoader() in zju_loader.py
```
data = self.fetch_data(index)
...
return self.preprocess(data)
```










