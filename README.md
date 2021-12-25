# RGANet
**Official Pytorch implementation for the journal article:**
```
    @article{mo2022rganet,  
        title = {Realtime Global Attention Network for Semantic Segmentation},
        author = {Mo, Xi and Chen, Xiangyu},
        journal = {IEEE Robotics and Automation Letters with ICRA presentation},
        year={2022}
    }
```  
<a href="https://drive.google.com/uc?export=view&id=1omq84eEIY5sjruHC3MZFUU0ZZ57fbaId"><img src="https://drive.google.com/uc?export=view&id=1omq84eEIY5sjruHC3MZFUU0ZZ57fbaId" style="width: 900px; max-width: 100%; height: auto" title="Click to enlarge picture" />
## Requirements
python >= 3.5  
pytorch >= 1.0.0  
**(optional)** [thop](https://github.com/Lyken17/pytorch-OpCounter), [apex](https://github.com/NVIDIA/apex), [tqdm](https://github.com/tqdm/tqdm)
## Demo
* Create the folder `checkpoint` in the root directory, [download](https://drive.google.com/file/d/1RhP3PK2sjW0Xsh0tOGGya_wz8EcOfOtV/view?usp=sharing) our pretrained checkpoint (53.1MB) to the folder, then run the demo:    
>```
> python demo.py
>```
>Suction area predictions should be saved in the folder `sample`, or specify the `path/to/checkpoint` and `path/to/samples` using `-c` and `-d` args respectively.
## Training
* Prepare dataset and train from scratch  
>Please consult `utils/configuraion.py` if you want to customize training setup, then download [suction-based-grasping-dataset.zip](https://vision.princeton.edu/projects/2017/arc/) (1.6GB), create the folder `dataset` in the root directory. There are two ways to train RGANet-NB:
>>Extract the main folder `suction-based-grasping-dataset` to `dataset`, run
>```
>(default) > python RGANet.py -train
>```
>>Extract the main folder to somewhere else and specify the paths:
>```
>(customized) > python RGANet.py -train -i path/to/color-input -l path/to/label
>```
* Restore training from checkpoint
>By default, RGANet read the latest checkpoint from the folder `checkpoint`, you can also specify the checkpoint using arg `-c`:
>```
> python RGANet.py -train -r -c path/to/checkpoint
>``` 
## Test
>The checkpoint is required before any test, and we only present the RGANet-NB architecture. Please consult `utils/configuraion.py` if you want to customize the testing, then run
>```
> python RGANet.py -test
>```
>Refer to arg `-d` to specify the path to save predictions. 
## Validation
>We provide both online validation and offline valuation. The online validation runs tests on all checkpoints and estimates the checkpoint that may has the best performance. Offline validation requires predictions saved to disk beforehand, i.e., run test w/ predictions written to disk before any offline validation. Please make sure you've set desired options in `utils/configuraion.py`. Run  
>```python RGANet.py -v``` or ```python RGANet.py -test -v```  
>Refer to arg `-d` to specify the path to predictions.
## Scripts
* In addition to the functions provided above, we also provide useful tools:

| name | illustration|
| :-----:| :----: |
| calculator.py | evaluation statistics, calculate model parameters | 
| eval_adaptor.py | split items of validation result to separate files | 
| models.py | standalone training, validation of segmentation models| 
| pred_transform.py | convert other predictions to processable images|
| proportion.py | compute adaptive weights for CE loss and focal loss|
| seg_models.py | standalone runtime test for segmentation models|
# License
Apache 2.0
