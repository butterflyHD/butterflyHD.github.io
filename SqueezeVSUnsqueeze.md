```python
import torch
```

Manipulating tensor dimensions is a common task that arises in various scenarios, from preprocessing data to interfacing with different model architectures. Two functions at the heart of dimension manipulation are squeeze() and unsqueeze(). Despite any initial impressions their names may give, these functions are pivotal for efficient data and model management. Let's delve into their usage and practical applications.

**The Essentials of `squeeze()` and `unsqueeze()`**
Both squeeze() and unsqueeze() are designed to modify the dimensions of tensors in specific ways:

unsqueeze(): This function **adds** a dimension of size one to a tensor at a specified index, increasing the tensor's dimensionality. It's particularly useful when an operation or model layer requires input data with certain dimensions.

squeeze(): In contrast, squeeze() **removes** dimensions of size one from a tensor's shape. It's often used to eliminate redundant dimensions that may complicate tensor handling or operations.

**Preparing Data for a Model**
Models, especially in domains like image processing or sequence modeling, may require input tensors with specific shapes. unsqueeze() can adjust your data to meet these requirements:


```python
x = torch.tensor([[1, 2],[3, 4]]) # 2x2 tensor 
print(x)
```

    tensor([[1, 2],
            [3, 4]])
    


```python
# add a dimension of size 1 at index 0 makes it 1x2x2 tensor 
print(x.unsqueeze(0)) # [ [ [1 ,2], [3, 4]  ] ]
```

    tensor([[[1, 2],
             [3, 4]]])
    


```python
print(x.unsqueeze(1)) # [ [ [1 ,2] ], [ [3, 4]  ] ]  2x1x2 tensor
```

    tensor([[[1, 2]],
    
            [[3, 4]]])
    


```python
print(x.unsqueeze(2)) # [ [ [1], [2] ], [ [3], [4]  ] ]  2x2x1 tensor
```

    tensor([[[1],
             [2]],
    
            [[3],
             [4]]])
    


```python
print(x.squeeze()) # it will remove all of the dimensions of size 1
```

    tensor([[1, 2],
            [3, 4]])
    
