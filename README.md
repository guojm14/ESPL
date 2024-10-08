# ESPL

Pytorch implementation of **Efficient Symbolic Policy Learning with Differentiable Symbolic Expression** [[paper]](https://arxiv.org/abs/2311.02104)

## CSP/  

The source code of **contextual symbolic policy**.

Train the contextual symbolic policy with:  

    python  main.py  --config configs/xxx.config   --spls 0.25 --target_ratio 0.002  --arch_index 0  --hard_epoch 25  --seed 0

You can tune the hyper-parameters by editing main.py or using the argparse.  

You can also get the average count of all selected paths and paths selected by at least ninety percent with CSP/mask_matrix.py

Then you can extract the discovered symbolic policy with CSP/sym.py (Please make sure the symbolic network is sparse enough before this). 




## ESPL/ 

The source code for **single-task symbolic policy learning**.

Train the symbolic policy with:

    python -u sac_symbolic_v1.py --env lunar_lander

You can tune the hyper-parameters by editing sac_symbolic_v1.py or using the argparse.

Then you can extract the discovered symbolic policy with ESPL/sym.py (Please make sure the symbolic network is sparse enough before this). 

## Example：

Video of symbolic policy playing car game:  
https://user-images.githubusercontent.com/12538710/226151704-9c4ddabe-d4c7-4b48-95c0-cea338142b2a.mp4




