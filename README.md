# DENet
An Universal Network for Counting Crowd with Varying Densities and Scales

Requirements: Python>3.4, Pytorch>=0.4.0 and tensorflow >=1.8

1.Use Colab to simplify your life 
2. Denet_first_etape
    2.1.compile mask rcnn based on
    2.2.Use Mask RCNN to detect and segment first
    2.3.Generate density maps
3.Denet_Second_etape:
  3.1.Trian the model with segmented image.
  3.2.Run validation 
