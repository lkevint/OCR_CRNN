This project is meant to explore the applicability of transfer learning using CRNN architecture pretrained on a clean, well labeled dataset like [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/), and fine-tuning it for better accuracy on messy datasets like [TextOCR](https://textvqa.org/textocr/) where the text examples are extremely inconsistent in size, shape, and font. The main paper we draw inspiration from is [OCR using CRNN: A Deep Learning Approach for Text (Yadav et al., 2023)](https://ieeexplore.ieee.org/document/10170436)

Preprocessing: Transform arbitrary quadrilateral labels to grayscaled rectangles
![Untitled](https://github.com/lkevint/OCR_CRNN/assets/68560628/a419b029-c6a8-45b5-a764-335d0c04743e)
