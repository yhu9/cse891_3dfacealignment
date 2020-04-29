# Bridging the Gap Between 3D and 2D FaceAlignment 

to run go to source.

for training with just the 3DMM run. This requires training data

  ```python train.py```
  
for training with both 3DMM and the adjustment run. This requires training data

  ```python train2.py```
  
for testing with just the 3DMM run without brackets. Input images must be preprocessed

  ```python test.py [imgname.png]```
  
for testing with both 3DMM and the adjustment run. Input images must be preprocessed

  ```python test2.py [imgname.png]```
  
for evaluation run

  ```python evaluate.py```
