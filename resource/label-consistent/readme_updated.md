1.首先将[链接](https://drive.google.com/drive/folders/1Qhj5vXX7kX74IWdrQDwguWsV8UvJmzF4)中的训练模型下载至文件夹`./resource/label-consistent`中（现已上传至仓库该文件夹中）
  <br/>
2.生成训练集与测试集，以`cifar10`为例，运行如下代码：
   ``` 
   python craft_adv_dataset.py --dataset cifar10 
   ```
   运行结束后`./resource/label-consistent/test/adv_dataset`文件夹下将出现`cifar10_train.npy`和`cifar10_test.npy`文件
   <br/>
3.进行lc-attack，以`cifar10`为例，运行如下代码：
   ```
   python ./attack/lc.py  --attack_train_replace_imgs_path ../resource/label-consistent/test/adv_dataset/cifar10_train.npy --attack_test_replace_imgs_path ../resource/label-consistent/test/adv_dataset/cifar10_test.npy --bd_yaml_path ../config/attack/lc/default.yaml --pratio 0.1 --save_folder_name lc_0_1
   ```
   运行前建议将相对路径修改为绝对路径
