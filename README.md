# SFINet
# Requirements
python 3.7 + pytorch 1.9.0 + imageio 2.22.2
# Saliency maps
We provide saliency maps of our SFINet on ORSSD, EORSSD and ORSI-4199 datasets.  
[SFINet-MobileNetV2](https://pan.baidu.com/s/129E-gxOyUlrENarZeTW8AQ) (code:SFIN)  
[SFINet-MobileNetV3](https://pan.baidu.com/s/1ARYS0Uun53FIRFbmrFHR0Q) (code:SFIN)  

# Training
Run train_SFINet.py.  
For SFINet-MobileNetV3, please modify paths of [MobileNetV3_backbone](https://pan.baidu.com/s/1aPX9yAaHtlbrSL5fP0HEfA) (code: SFIN) in ./model/SFINet_V3.py.  
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/SFINet/, then run test_SFINet.py.  
[SFINet_V2_EORSSD](https://pan.baidu.com/s/1e0hjuF1ENcxHXay5Rkaf-A) (code:SFIN)  
[SFINet_V2_ORSSD](https://pan.baidu.com/s/1TMo4OZXwgiYoi8crK3GulA) (code:SFIN)  
[SFINet_V2_ORSI-4199](https://pan.baidu.com/s/1jCvlBXCEsyjKTYNMdw9cGw) (code:SFIN)  
[SFINet_V3_EORSSD](https://pan.baidu.com/s/1P0hdffO1WZyBV53bmJ3POA) (code:SFIN)  
[SFINet_V3_ORSSD](https://pan.baidu.com/s/1XBdfKea016IoNr_-kHXwzQ) (code:SFIN)  
[SFINet_V3_ORSI-4199](https://pan.baidu.com/s/1C2q3sJ_0VeehspQPGBeCSQ) (code:SFIN)  

# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
