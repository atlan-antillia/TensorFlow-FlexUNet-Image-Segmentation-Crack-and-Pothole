<h2>TensorFlow-FlexUNet-Image-Segmentation-Crack-and-Pothole (2025/11/09)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Crack and Pothole in Road</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 768x480(475) pixels  
<a href="https://drive.google.com/file/d/1ZkSKNJYwhxfUspvf8K2RcT-5TL5mcIbV/view?usp=sharing">
<b>Augmented-Crack-Pothole-ImageMask-Dataset.zip</b></a>  with colorized masks,  
which was derived by us from 
<a href="https://data.mendeley.com/datasets/t576ydh9v8/4">
<b>Cracks and Potholes in Road Images</b>
</a>
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of <b>Crack and Pothole</b>, which contains 2235 images and labels respectively,
we used our offline augmentation tools <a href="https://github.com/sarah-antillia/Image-Distortion-Tool"> 
Image-Distortion-Tool</a>
and
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a> 
 to augment the original dataset.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of  1024x630 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
Augmented dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map(crack:green, pothole: red)</b><br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/1430.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/1430.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/1430.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/1425.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/1425.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/1425.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/3140.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/3140.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/3140.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from 
<br><br>
<a href="https://data.mendeley.com/datasets/t576ydh9v8/4">
<b>Cracks and Potholes in Road Images</b>
</a>
<br>
<br>
Please see also 
<a href="https://biankatpas.github.io/Cracks-and-Potholes-in-Road-Images-Dataset/">
Cracks-and-Potholes-in-Road-Images-Dataset
</a>
<br>
<br>
<b>Citation</b><br>
Passos, Bianka T.; Cassaniga, Mateus J.; Fernandes, Anita M. R. ; Medeiros, Kátya B. ; Comunello, Eros (2020), <br>
“Cracks and Potholes in Road Images”, Mendeley Data, V4, doi: 10.17632/t576ydh9v8.4
<br>
<br>
<b>Description</b><br>
This database contains images of defects in paved roads in Brazil. The database was developed using images provided by DNIT (National Department of Transport Infrastructure), through the Access to Information Law - Protocol 50650.003556 / 2017-28.
<br>
The database contains:<br>
* 2235 images from highways in the states of Espirito Santo, Rio Grande do Sul and the Federal District, caught between 2014 and 2017; and
<br>
* Each image has 3 masks - binary images in png format - separated for each type of annotation: road, crack and pothole. 
<br>
The annotation of the road consisted of demarcating the total region corresponding to the vehicle's road. 
<br>
The annotation of cracks and potholes consisted of the selection of the defect as a whole, maintaining its shape as much as possible.

<br>
<br>
<b>LICENSE</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
<br>
<br>

<h3>
2 Crack-Pothole ImageMask Dataset
</h3>
 If you would like to train this Crack-Pothole Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1ZkSKNJYwhxfUspvf8K2RcT-5TL5mcIbV/view?usp=sharing">
 <b>Augmented-Crack-Pothole-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Crack-Pothole
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Crack-Pothole Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Crack-Pothole/Crack-Pothole_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
Each colorized mask of our dataset was generated by combining a crack mask and a pothole mask corresponding to a raw image, 
and colorizing the masks with a color-map (crack:green, pothole:red), as shown below.<br>
<br>
<table>
<tr><th>raw image</th><th>crack mask</th><th>pothole mask</th><th>colorized mask</th></tr>
<tr>
<td><img src = "./projects/TensorFlowFlexUNet/Crack-Pothole/asset/1014889_RS_386_386RS124739_31370_RAW.jpg" width="250" height="auto"></td>
<td><img src = "./projects/TensorFlowFlexUNet/Crack-Pothole/asset/1014889_RS_386_386RS124739_31370_CRACK.png" width="250" height="auto"></td>
<td><img src = "./projects/TensorFlowFlexUNet/Crack-Pothole/asset/1014889_RS_386_386RS124739_31370_POTHOLE.png"  width="250" height="auto"></td>
<td><img src = "./projects/TensorFlowFlexUNet/Crack-Pothole/asset/1014889_RS_386_386RS124739_31370.png"  width="250" height="auto"></td>
</tr>
</table> 
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Crack-Pothole TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Crack-Pothole/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Crack-Pothole and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 3

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Crack-Pothole 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                   crack:green, pothole;red,   
rgb_map = {(0,0,0):0,(0,255,0):1, (255,0, 0):2, }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 26,27,28,29)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 52,53,54,55)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 55.<br><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/train_console_output_at_epoch55.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Crack-Pothole/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Crack-Pothole/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Crack-Pothole</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Crack-Pothole.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/evaluate_console_output_at_epoch55.png" width="720" height="auto">
<br><br>Image-Segmentation-Crack-Pothole

<a href="./projects/TensorFlowFlexUNet/Crack-Pothole/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Crack-Pothole/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0203
dice_coef_multiclass,0.9892
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Crack-Pothole</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Crack-Pothole.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Crack-Pothole/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 1024x630 pixels </b><br>
<b>rgb_map(crack:green, pothole: red)</b><br>

<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/1425.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/1425.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/1425.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/1427.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/1427.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/1427.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/1433.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/1433.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/1433.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/3137.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/3137.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/3137.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/3141.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/3141.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/3141.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/images/3144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test/masks/3144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack-Pothole/mini_test_output/3144.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Cracks-and-Potholes-in-Road-Images-Dataset</b><br>
Bianka Tallita Passos, Mateus Junior Cassaniga, Anita Maria da Rocha Fernandes, Kátya Balvedi Medeiros, Eros Comunello<br>
<a href="https://biankatpas.github.io/Cracks-and-Potholes-in-Road-Images-Dataset/">
https://biankatpas.github.io/Cracks-and-Potholes-in-Road-Images-Dataset/
</a>

<br>
<br>
<b>2. We Learn Better Road Pothole Detection:<br>
from Attention Aggregation to Adversarial Domain Adaptation</b><br>
Rui Fan1, Hengli Wang, Mohammud J. Bocus, and Ming Liu<br>
<a href="https://arxiv.org/pdf/2008.06840">https://arxiv.org/pdf/2008.06840</a>
<br>
<br>
<b>3. Deep transformer networks for precise pothole segmentation tasks</b><br>
Iason Katsamenis, Athanasios Sakelliou, Nikolaos Bakalos, Eftychios Protopapadakis, Christos Klaridopoulos,<br>
Nikolaos Frangakis, Matthaios Bimpas, Dimitris Kalogeras<br>
<a href="https://dl.acm.org/doi/fullHtml/10.1145/3594806.3596560">https://dl.acm.org/doi/fullHtml/10.1145/3594806.3596560</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Crack </b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Crack">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Crack</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Pothole </b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Pothole">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Pothole</a>
<br>
<br>
