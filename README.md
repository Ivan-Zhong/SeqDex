# Lego Project with RealMan Inspire

## RL Policy Training

### Search

观测空间中，和物块相关的feature包括一个二值0/1的“能否看到这个物块”（一定要大于一定的阈值才能算看到），以及物块的pos和rot（能看到的时候是真实值，不能看到的时候给-1），然后去把它给完全地刨出来，直到完全露出。

reward是希望上面没有物块，没有物块reward是0，有一个物块就减少一些reward
reward shaping：手离物块要在一定范围内；手的姿态要好；希望物块要不断的在动

因此总的reward是：
如果没看到物块，手的姿态要好，希望物块在不断移动；
看到物块的时候，手的姿态要好，要离物块比较近，希望物块上面覆盖的物块比较少。


### Grasp


## Sim2Real