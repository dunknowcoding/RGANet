# RGANet
**Official Pytorch implementation for the journal article:**
```
    @article{mo2021rganet,  
        title = {Realtime Global Attention Network for Semantic Segmentation},
        author = {Mo, Xi and Chen, Xiangyu},
        journal = {IEEE Robotics and Automation Letters with ICRA presentation},
        year={2022}
    }
```
## Requirements
python >= 3.5  
pytorch >= 1.0.0  
**(optional)** [thop](https://github.com/Lyken17/pytorch-OpCounter), [apex](https://github.com/NVIDIA/apex)
## Demo
* Create the folder `checkpoint` in the root directory, [download](https://drive.google.com/file/d/1RhP3PK2sjW0Xsh0tOGGya_wz8EcOfOtV/view?usp=sharing) our pretrained checkcpoint to the folder, then run the demo:    
```
> python demo.py
```
Suction area predictions will appears in the folder `sample`. 