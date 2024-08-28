# **Conditional TGAN with Singular Valued Clipping** 
This repository contains an implementation of a Conditional Temporal Generative Adversarial Network (TGAN) model using singular value clipping. The model is designed to generate sequences conditioned on text inputs. It follows the principles from the base TGAN paper and integrates text embeddings to guide the generation process.
## **Overview**
1) **Project:** Conditional TGAN (Text-Guided Temporal GAN)
2) Singular value clipping for enforcing the 1-Lipschitz constraint.
3) Checkpoint saving and loading functionality.
4) Drawing inferences on the saved generator
## **Geting started** 
Get the repository in your local machine by following the below command 
```bash
git clone https://github.com/origin459/Conditional_TGAN.git
``` 
## **Requirements** 
```bash
pip install -r requirements.txt
```
## **Dataset**
The dataset path should be specified in main.py. There is an example dataset given in the repo by the name **"Fireball_animations"**. Refer to that dataset and make sure there are enough training samples.
## **Training** 
To train the model, execute the training script. The training configuration, including dataset paths and save periods, is specified in **main.py**.  
```bash
python main.py
```
## **Parameters** 
- **Dataset Path**: Define the dataset path in **main.py**.
* **Save Period**: Define how frequently the model checkpoints should be saved. This is in **main.py**. The saved instances are saved in the folder called **saves**.
## **Usage** 
After the model has been saved in the **saves** folder, run the following command: 
```bash
python inference.py
```
The **inference.py** takes a label parameter that can be specified in the first line of the same file in a variable called **label**. Change it accordingly to generate a robust animation even with unseen yer similar labels. The **inference.py** uses the latest saved model in the folder **saves**. 
The output of the generator is good for low resoltion short videos. This model does not scale up to generate higher quality videos. But in case of low resolution GIF the model can sufficiently generate convincing outputs by taking labels.
## **Acknowledgement** 
This model is inspired from the base paper on TGAN that showed the usage of a GAN model for video generation. 
### **Citations** 
``` bash
@inproceedings{TGAN2017,
    author = {Saito, Masaki and Matsumoto, Eiichi and Saito, Shunta},
    title = {Temporal Generative Adversarial Nets with Singular Value Clipping},
    booktitle = {ICCV},
    year = {2017},
}
```
