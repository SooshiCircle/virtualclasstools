# Virtul Learning Toolbox
## Project Overview

It is not surprising, given the ongoing global pandemic, that the demand is high for remote learning and connectivity programs.
As such, our team proposes a set of tools for virtual interaction that may better facilitate global communication during these solitary times.

Firstly, we created a hand-controlled drawing program.
This allows for the user to move their hand and draw directly on their video feed. We implemented a hand-detection system by [Victor Dibidia](https://github.com/victordibia/handtracking) to detect hands in the video feed.
Using OpenCV, we allow the user to toggle drawing mode “on” and draw over their video feed by detecting the corner of their hand.
To improve performance, we created a filter to remove sharp jumps and restrict detection to just the user’s hand.
This tool will help teachers create quick visualizations on their video feed to aid explanations and reduce the distraction of finding a medium through which to share visual diagrams.

Our second tool is vocal response polling.
Students can verbally answer a professor’s question and the program will analyze their individual responses.
We used [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) API for speech to text.
We used TF-IDF values to summarize the most important words of  response.
We also used GloVe embeddings to compute the similarity between student responses with the intention of forming appropriate breakout rooms.

Our third tool is a hand raised detection feature.
Our original idea was to train a ResNet to classify between a raised hand and thumbs up on a camera feed.
Our progress includes preliminary training of an untrained [ResNet](https://github.com/timesler/facenet-pytorch) using PyTorch, gathering our own data by taking pictures of various hand gestures, and experimenting with training parameters to try to improve performance.
Unfortunately, due to the tight time constraints of the competition, we were not able to implement a well-trained model that accurately differentiates these hand gestures.
Instead, we relied on mathematical methods, where we calculated cluster average points and hand recognition to detect when a student is raising their hand during a live video feed.

This tool will help alert teachers to students who physically raise their hand, bringing more of a real classroom feel to a virtual environment.
With our virtual learning toolbox, we hope to provide access to many new and useful features in the realm of remote communication.

## Demos and Presentation
For a graphical explanation of our project and live demos of each feature, please see our PowerPoint presentation, which can be found [here](https://drive.google.com/file/d/1Fj20_z3JTIZ620n88CUMj0fgCvyIIPBN/view?usp=sharing).
