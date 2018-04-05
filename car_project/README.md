## Car Detection

#### `car_detection_v1.ipynb` : the simplest plain **CNN** model on **CIFAR-10 datasets**.

* data : the **positive** and **negative** images are both **6000**. The shape of image is **(32,32,3)**.

<div align="center">
  <img src="images/CIFAR_cars.png" height="400" width="500"><br>             
</div>

* model : Input --> (CONV->AVG_POOL->ReLU) x 4 --> (NN->ReLU) x 3 --> (NN->Sigmoid) --> Output.
