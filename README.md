
# Cross modal facial image synthesis using a collaborative bidirectional style transfer network


NIZAM UD DIN1, SEHO BAE
, KAMRAN JAVED
, AND JUNEHO YI



The goal of this research is to synthesize realistic cross modal face images while retaining the input face identity.


![image](https://user-images.githubusercontent.com/27881319/171996834-788745d9-def8-4c90-8e4a-a100b9808ba9.png)

[CUFS](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)

For the task of segmentation ⇄ photo synthesis:

we use the CelebAMask-HQ dataset. We use the photo/sketch paired [CUFS](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html) dataset for photo ⇄ sketch synthesis task.

For sketch ⇄ segmentation synthesis task:

we have created a dataset for color coded segmentation map and their corresponding sketches using the publicly available photo/sketch paired dataset ([CUFS](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)). 

To achieve this, we use the model trained for the photo ⇄ segmentation synthesis task. We translate all photos from the [CUFS](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html) dataset into segmentation map and use those synthesized segmentation maps along with the corresponding sketches as paired segmentation/sketch samples 

![image](https://user-images.githubusercontent.com/27881319/172046393-5b76741a-0191-4e3d-906e-bf2565186f49.png)
