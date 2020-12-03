# GAN-Image-Super-Resolution
* Part of a **PROJECT**
* **Image Super-Resolution Using a Generative Adversarial Network** (**SR-GAN**)
* Tensorflow 2.3

### SR-GAN Architecture

<img src="samples/results/model.jpeg" width="800"/>

### Train on your own Data

 Fork the repository.
 
 
 Dump the HR(**High-Resolution**)image under `Data/HR/` and LR(**Low-Resolution**)image under `Data/LR/`.
 
Make sure about 
        
        HR Images (Totall sample, 96*4, 96*4, 3)
        LR Images (Totall sample, 96, 96, 3)
        
Run the following in command prompt in current dir for traning
        
        python train.py

Model will get saved in `checkpoint` folder in running EPOCHS.


### Results

<img src="samples/results/inp_LR.png" width="200"/> LRI <img src="samples/results/predict_HR.png" width="200"/> HRP <img src="samples/results/ref_HR.png" width="200"/>  HRO

<br>

<img src="samples/results/inp_LR1.png" width="200"/> LRI <img src="samples/results/predict_HR1.png" width="200"/> HRP <img src="samples/results/ref_HR1.png" width="200"/>  HRO



### References
1. **SR-GAN**- https://github.com/tensorlayer/srgan
2. **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**- https://arxiv.org/pdf/1609.04802.pdf
3. **SUPER-RESOLUTION WITH DEEP CONVOLUTIONAL SUFFICIENT STATISTICS**- https://arxiv.org/pdf/1511.05666.pdf
