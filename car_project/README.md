# Car Detection

### `car_detection_v1.ipynb` : the simplest plain CNN model on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.<br><br>

* data : the **positive** and **negative** images are both **6000**. The shape of image is **(32,32,3)**. Batch size is **64**. <br>

<div align="center">
  <img src="images/version1/CIFAR_cars.png" height="400" width="500" /><br><br>            
</div>

* model : input --> (conv->avg_pool->relu) **x 4** --> (nn->relu) **x 3** --> (nn->sigmoid) --> output.

<div align="center">
  <img src="images/version1/model.png" height="320" width="560" /><br>             
</div>

* costs : **third** epcho(left) and **fourth** epoch(right)

<div align="center">
  <img src="images/version1/3.png" height="240" width="360"/>
  <img src="images/version1/4.png" height="240" width="360"/>
</div>
