# CL4SRec-pytorch
A pytorch implementation of CL4SRec in "Contrastive Learning for Sequential Recommendation", which provides three output aggregation strategies including 'concat', 'mean' and 'predict' and three augmentation strategies 'mask', 'reorder' and 'crop'. 

## Dataset
The dataset should be organized as the following format. The first column is the userid, followed by the interacted items.

```python
# ./data/dataset_name.txt
user item1 item2 ...
```

## Usage
You can train CL4SRec on Yelp dataset by following command
```bash
python -u main.py --dataset Yelp --cl_embs predict
```

## Acknowledgement
The Transformer layer is implemented based on [recbole](https://github.com/RUCAIBox/RecBole).