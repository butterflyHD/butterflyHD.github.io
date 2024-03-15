---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 2
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython2
    version: 2.7.6
  nbformat: 4
  nbformat_minor: 5
---

::: {#initial_id .cell .code execution_count="8" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:37.360248Z\",\"start_time\":\"2024-03-14T06:13:37.336572Z\"}" collapsed="true"}
``` python
import torch
```
:::

::: {#ba0f912aba74cc64 .cell .markdown collapsed="false"}
Manipulating tensor dimensions is a common task that arises in various
scenarios, from preprocessing data to interfacing with different model
architectures. Two functions at the heart of dimension manipulation are
squeeze() and unsqueeze(). Despite any initial impressions their names
may give, these functions are pivotal for efficient data and model
management. Let\'s delve into their usage and practical applications.

**The Essentials of `squeeze()` and `unsqueeze()`** Both squeeze() and
unsqueeze() are designed to modify the dimensions of tensors in specific
ways:

unsqueeze(): This function **adds** a dimension of size one to a tensor
at a specified index, increasing the tensor\'s dimensionality. It\'s
particularly useful when an operation or model layer requires input data
with certain dimensions.

squeeze(): In contrast, squeeze() **removes** dimensions of size one
from a tensor\'s shape. It\'s often used to eliminate redundant
dimensions that may complicate tensor handling or operations.

**Preparing Data for a Model** Models, especially in domains like image
processing or sequence modeling, may require input tensors with specific
shapes. unsqueeze() can adjust your data to meet these requirements:
:::

::: {#50d5f12a78e0d466 .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:39.888763Z\",\"start_time\":\"2024-03-14T06:13:39.883776Z\"}" collapsed="false"}
``` python
x = torch.tensor([[1, 2],[3, 4]]) # 2x2 tensor 
print(x)
```

::: {.output .stream .stdout}
    tensor([[1, 2],
            [3, 4]])
:::
:::

::: {#cd2192e44905e288 .cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:40.571569Z\",\"start_time\":\"2024-03-14T06:13:40.564180Z\"}" collapsed="false"}
``` python
# add a dimension of size 1 at index 0 makes it 1x2x2 tensor 
print(x.unsqueeze(0)) # [ [ [1 ,2], [3, 4]  ] ]
```

::: {.output .stream .stdout}
    tensor([[[1, 2],
             [3, 4]]])
:::
:::

::: {#79c274f9057aeb99 .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:41.067391Z\",\"start_time\":\"2024-03-14T06:13:41.065108Z\"}" collapsed="false"}
``` python
print(x.unsqueeze(1)) # [ [ [1 ,2] ], [ [3, 4]  ] ]  2x1x2 tensor
```

::: {.output .stream .stdout}
    tensor([[[1, 2]],

            [[3, 4]]])
:::
:::

::: {#e04be5ab5e59c1fe .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:41.691927Z\",\"start_time\":\"2024-03-14T06:13:41.687968Z\"}" collapsed="false"}
``` python
print(x.unsqueeze(2)) # [ [ [1], [2] ], [ [3], [4]  ] ]  2x2x1 tensor
```

::: {.output .stream .stdout}
    tensor([[[1],
             [2]],

            [[3],
             [4]]])
:::
:::

::: {#8a657e00f907919d .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-03-14T06:13:42.184930Z\",\"start_time\":\"2024-03-14T06:13:42.181335Z\"}" collapsed="false"}
``` python
print(x.squeeze()) # it will remove all of the dimensions of size 1
```

::: {.output .stream .stdout}
    tensor([[1, 2],
            [3, 4]])
:::
:::
