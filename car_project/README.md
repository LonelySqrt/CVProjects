# Car Detection

### `car_detection_v1.ipynb` : the simplest plain CNN model on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.<br><br>

* data : the **positive** and **negative** images are **both 6000**. The shape of image is **(32,32,3)**. Batch size is **64**. <br>
<div align="center">
  <img src="images/version1/CIFAR_cars.png" height="400" width="500" /><br>            
</div>

----

* conv-block : 
<div align="center">
  <img src="images/version1/conv-block.png" height="240" width="740" /><br>             
</div>

* model :
<div align="center">
  <img src="images/version1/model.png" height="190" width="800" /><br>             
</div>

----

* cost : the **first** epoch. **NOTE THAT: I haven't added any normalization yet.**<br>

<div align="center">
  <img src="images/version1/epoch1-cost.png" height="250" width="370"/>
</div>

<div align="center">
  <p> Generally you could see the cost is descenting slowly, and there is a lot <strong>oscillation</strong> because of mini-batch. </p>
</div>
