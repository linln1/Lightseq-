## 总体分析

- layer A 

- layer B 

各有26层，也就是这个Transformer共有52个Encoder





分析两个文件

## 1.先分析这个类的内存占用情况

- TransformerEncoderLayer类

- 下载 lightseq项目，创建visual studio CUDA项目，在编译选项命令行中添加

- ```
  /d1 reportAllClassLayout
  ```

- 然后编译transformer_encoder_layer.cpp, 得到TransformerEncoderLayer类的内存分布， 是488个字节

- ```
  +    class TransformerEncoderLayer<float>	size(488):
  +    	+---
  +     0	| {vfptr}
  +     4	| _layer_id
  +     8	| _H
  +    12	| _heads
  +    16	| _intermediate_size
  +    20	| _max_batch_tokens
  +    24	| _max_seq_len
  +    28	| _pre_or_postLayerNorm
  +      	| <alignment member> (size=3)
  +    32	| ?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@ _activation_fn
  +    60	| _batch_size
  +    64	| _seq_len
  +    68	| _batch_tokens
  +    72	| _batch_heads
  +    76	| _batch_dim
  +    80	| _training
  +      	| <alignment member> (size=3)
  +    84	| _cublasHandle
  +    88	| _stream
  +    92	| ?$FeedForward@M _qkv_linear
  +    112	| ?$FeedForward@M _attn_out_linear
  +    132	| ?$Normalize_Layer@M _attn_ln
  +    148	| ?$Normalize_Layer@M _ffn_ln
  +    164	| ?$FeedForward@M _ff1
  +    184	| ?$FeedForward@M _ff2
  +    204	| ?$Softmax@M _softmax
  +    208	| ?$Dropout@M _attn_prob_dropout
  +    224	| ?$Dropout@M _attn_dropout
  +    240	| ?$Dropout@M _ffn_activation_dropout
  +    256	| ?$Dropout@M _ffn_dropout
  +    272	| ?$StridedBatchGemm@M _attn_scores
  +    316	| ?$StridedBatchGemm@M _attn_context
  +    360	| _gemmQKV_inp_ptr
  +    364	| _qkv_ptr
  +    368	| _soft_out_ptr
  +    372	| _ctx_bufB_ptr
  +    376	| _attn_o_inp_ptr
  +    380	| _ff1_inp_ptr
  +    384	| _relu_inp_ptr
  +    388	| _ff2_inp_ptr
  +    392	| _attn_qkvw_ptr
  +    396	| _attn_qkvb_ptr
  +    400	| _attn_ow_ptr
  +    404	| _attn_ob_ptr
  +    408	| _attn_nw_ptr
  +    412	| _attn_nb_ptr
  +    416	| _inter_w_ptr
  +    420	| _inter_b_ptr
  +    424	| _output_w_ptr
  +    428	| _output_b_ptr
  +    432	| _ffn_nw_ptr
  +    436	| _ffn_nb_ptr
  +    440	| _grad_attn_qkvw_ptr
  +    444	| _grad_attn_qkvb_ptr
  +    448	| _grad_attn_ow_ptr
  +    452	| _grad_attn_ob_ptr
  +    456	| _grad_attn_nw_ptr
  +    460	| _grad_attn_nb_ptr
  +    464	| _grad_inter_w_ptr
  +    468	| _grad_inter_b_ptr
  +    472	| _grad_output_w_ptr
  +    476	| _grad_output_b_ptr
  +    480	| _grad_ffn_nw_ptr
  +    484	| _grad_ffn_nb_ptr
  ```

- 仔细分析如下

  ```C
  【1】
  0字节-3字节是class TransformerEncoderLayer类虚函数表指针的空间
  【2】
  4字节-31字节是EncoderLayer的一些常量参数，都是const修饰的常量，一般是初始化给定
  【3】
  32-59字节： string
  【4】
  60-83字节是一些EncoderLayer的变量
  【5】
  84-91字节是两个cublas结构体
  【6】
  92-131字节是两个前向传播层的结构体
  【7】
  132-163字节是两个正则化层Normalize_Layer的结构体
  【8】
  164-203是两个前向传播层的结构
  【9】
  204-207字节是Softmax层
  【10】
  208-271字节是4个Dropout层
  【11】
  272-359字节是2个矩阵乘法层
  【12】
  360-391GPU本地内存，8个指针，分别代表层间的训练权重的指针，共32字节
  【13】然后注意到类中有一个静态变量，是层与层之间共享的变量，放在shared GPU mem中，猜想对应于GPU硬件中的共享内存区域
    // shared GPU memory between layer
    static T *_shared_mem_ptr;
  【14】
  392-439字节，用于存放EncoderLayer中的参数指针，LayerA中用到
  【15】
  440-484字节，用于存放EncoderLayer中后向传播的梯度，LayerB中用到
   
  ```

- 然后看下分为几种类别

  - ![image-20211013005016726](encoderlayer.assets/image-20211013005016726.png)

  - ![image-20211013005047687](encoderlayer.assets/image-20211013005047687.png)

    
  
  - 
  
- 接着来看，什么是中间值，就是指运行过程中会动态分配内存的量

  - 中间值也就是

    - allocate_mem_buffer()
    - free_mem_buffer()

    中涉及到的变量指针

  - - 具体来看，内存中分配空间大小如下

      - ```
        _gemmQKV_inp_ptr:   _max_batch_tokens * _H 
        				  = _max_batch_size * _max_seq_len * _H
        
        _qkv_ptr: _max_batch_tokens * _H * 3 【用来生成KVQ矩阵】
        
        _soft_out_ptr: _max_batch_tokens * _heads * _max_seq_len 									用在【soft(KQ/sqrt(d_k))】
        
        _ctx_bufB_ptr: _max_batch_tokens * _heads * _max_seq_len
        
        _attn_o_inp_ptr = _max_batch_tokens * _H
        
        _ff1_inp_ptr = _max_batch_tokens * _H
        
        _relu_inp_ptr = _max_batch_tokens * _intermediate_size
        
        _ff2_inp_ptr = _max_batch_tokens * _intermediate_size
        ```

      - 如果层与层之间要共享gpu内存，就需要给__shared_mem_ptr分配空间smem_size

        - ```
          
          // buffer size needed by ffn bw
          size_t sz_ffn_bw = 3 * _max_batch_tokens * _H +
                                 _max_batch_tokens * _intermediate_size;
          
          
          
          // buffer size needed by attn bw
          size_t sz_attn_bw = 5 * _max_batch_tokens * _H +
                                  std::max(3 * _max_batch_tokens * _H,
                                           _max_batch_tokens * _heads * _max_seq_len);
                                           
                                           
              
          size_t smem_size = std::max(sz_ffn_bw, sz_attn_bw);
          ```

## 2. 分析任务中的52个Layer，运⾏起来⼤概会占⽤多少显存？

![image-20211013020206923](encoderlayer.assets/image-20211013020206923.png)

layer a与layer b 计算方法相同

显存占用就是   

- cuda_malloc中分配的量
- 计算过程中可能存在的临时变量

-  \__device__修饰的量
-  \__global__修饰量 



## 2.1 cuda_malloc 分配的量

	为了便于查看，我们做如下约定
	max_batch_tokens = B
	H = H
	max_seq_len = L
	heads = N
	

- ```
  
  - wptr = gptr 
  
  	   = 3 * H * H + 3 * H + H * H + H + H + H + H * intermediate_size + intermediate_size + H * intermediate_size + H + H + H 
  	   
  	   = 9 * H + 4 * H * H + 2 * H * intermediate_size + intermediate_size
  
  - qkv_ptr= 3 * B * H
  
  - soft_out_ptr = B * N * L
  
  - ctx_bufB_ptr = B * N * L
  
  - attn_o_inp_ptr = B * H
  
  - ff1_inp_ptr = B * H
  
  - relu_inp_ptr = B * intermediate_size
  
  - ff2_inp_ptr = B * intermediate_size
  
  - smem_size = max(3 * B * H + B * _intermediate_size, 5 * B * H + std::max(3 * B * H,  B * N * L))
  
  
  ```

​    其中smem_size是gpu中的全局共享内存，就是一些前后顺序上互相不依赖的tensor，可以共享显存。



## 2.2 计算过程中可能存在的临时变量

- 前向传播中

  - ![image-20211014212129197](encoderlayer.assets/image-20211014212129197.png)
  - 每一个TransformerEncoderLayer可以划分为这些子层，在运算过程中会产生一些显存占用
  - 

- 后向传播中

  - lightseq论文作者发现，transformer在训练和推理过程中，pytorch、tensorflow版代码gpu会频繁地释放和申请显存空间，导致gpu的显存利用率出现波动，如图：

    ![image-20211014205805248](encoderlayer.assets/image-20211014205805248.png)

    ​	所以lightseq针对此做了一些改进，在训练之前，预先估计gpu显存峰值大小，固定分配这么大的显存，然后就节约了训练过程中显存的动态申请和释放开销，举例来说，就是反向梯度传播中，同一列的数值可以共享显存，总共的显存占用就是3BLH + max{BLLN, 3BLN}

    ![image-20211014205950825](encoderlayer.assets/image-20211014205950825.png)

    

    - 右边部分的图的每一行列出了一步中临时张量的内存占用情况。
    - 同一列中的张量重用同一内存块。
    - 橙色张量和紫色张量的大小分别为BLH和BLLN。
    - 虚线内存块中的张量不会在这个步骤中更新，而实心内存块中的张量会更新。
    - 我们可以看到，只需要3BHL(前三个块)+ max{3BHL, BNL^2}(最后一个块)的内存字节.相反，如果不使用共享内存块策略，则总共需要9BLH+BL2N字节的内存。



## 3.再分析有52各这⼏类内存如何优化？

- **显卡内存可以分为**
  - 固定大小的永久内存
  - 存储参数及其梯度
  - 可变大小的临时内存来存储中间状态。



- 可以共享一些显存，理由是

  

- 共享需要一些前提条件

- 我们通过压缩内存来减少分配和释放，并且没有额外的成本，从而减少内存占用。

  

- 为了避免临时内存的频繁分配，我们对训练集进行扫描并估计其容量的上界。因此，在训练开始前分配一次大小最大的临时内存，并对不同批次进行重复使用，在训练结束后释放。具体方式就是第二问提到的。

