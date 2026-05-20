import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
    logit:(batch,seq,vocab)
    target:(batch,seq)

    l=-log(softmax(o_y))=-log(exp(o_y)/sum(exp(o_j)))
    =log(sum(exp(oj)))-o_y
    =log_sum_exp-o_y
    =M+log(sum(exp(o_j-M)))-o_y
    M meads max
    """

    # 1. get the max called M
    m=torch.max(logits,dim=-1,keepdim=True).values
    # 2. get the origin o_y
    target_logits=torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    # 3. calculate Log-Sum-Exp
    shift_logits=logits-m
    log_sum_exp=m.squeeze(-1)+torch.log(torch.sum(torch.exp(shift_logits),dim=-1))
    # 4. calculate each Token loss
    loss=log_sum_exp-target_logits
    # 5. calculate average and get a int 
    return loss