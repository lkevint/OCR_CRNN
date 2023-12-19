This project is meant to explore the applicability of transfer learning using CRNN architecture pretrained on a clean, well labeled dataset like [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/), and fine-tuning it for better accuracy on messy datasets like [TextOCR](https://textvqa.org/textocr/) where the text examples are extremely inconsistent in size, shape, and font. The main paper we draw inspiration from is [OCR using CRNN: A Deep Learning Approach for Text (Yadav et al., 2023)](https://ieeexplore.ieee.org/document/10170436), as it is recent and achieved outstanding accuracy on benchmark datasets (not TextOCR). Our TensorFlow implementation is very similar to [this tutorial](https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras/blob/master/CRNN%20Model.ipynb), which was based off of [this paper](https://arxiv.org/pdf/1507.05717.pdf), and we additionally looked at [this repository](https://github.com/GitYCC/crnn-pytorch/tree/master/src)

Preprocessing: Transform arbitrary quadrilateral labels to grayscaled rectangles
![Untitled](https://github.com/lkevint/OCR_CRNN/assets/68560628/a419b029-c6a8-45b5-a764-335d0c04743e)


MJSynth training/validation loss per epoch: 
![image](https://github.com/lkevint/OCR_CRNN/assets/68560628/88cc8c5c-30c0-4678-98d0-fcf7e88f9bcc)

Results on TextOCR:

