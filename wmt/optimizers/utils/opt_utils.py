import math
import torch


def _adam_step(p, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    grad = p.grad

    if len(state) == 0:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p)
        state['exp_avg_sq'] = torch.zeros_like(p)

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    state['step'] += 1
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    denom = exp_avg_sq.sqrt().add_(eps)

    bias_correction1 = 1.0 - beta1 ** state['step']
    bias_correction2 = 1.0 - beta2 ** state['step']
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    p.data.addcdiv_(exp_avg, denom, value=-step_size)


def adam_step(optimizer, layer_info, lr, betas, eps):
    beta1, beta2 = betas

    for pname, p in layer_info['params']:
        state = optimizer.state[p]
        if p.grad is None:
            continue

        _adam_step(p, state, lr, beta1, beta2, eps)


def momentum_step(optimizer, layer_info, step_size, momentum, weight_decay,
                  apply_wdecay=True):
    # update parameters
    for _, p in layer_info['params']:
        d_p = p.grad.data

        if weight_decay != 0 and apply_wdecay:
            d_p.add_(p.data, alpha=weight_decay)

        if momentum != 0:
            param_state = optimizer.state[p]
            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.zeros_like(p)

            d_p = param_state['momentum_buffer'].mul_(momentum).add_(d_p)

        p.data.add_(d_p, alpha=-step_size)


def nag_step(optimizer, layer_info, lr, momentum, weight_decay,
             apply_wdecay=True):
    for _, p in layer_info['params']:
        d_p = p.grad.data

        if weight_decay != 0 and apply_wdecay:
            d_p.add_(p.data, alpha=weight_decay)
        
        #if weight_decay != 0 and apply_wdecay and no_prox == True:
        #    p.data.mul_(1-lr*weight_decay)
        
        if momentum != 0:
            param_state = optimizer.state[p]
            if 'momentum_buff' not in param_state:
                param_state['momentum_buff'] = d_p.clone()
            else:
                buf = param_state['momentum_buff']
                buf.mul_(momentum).add_(d_p)
                d_p.add_(buf, alpha=momentum)

        p.data.add_(d_p, alpha=-lr)
        
        #if weight_decay != 0 and apply_wdecay and no_prox == False:
        #    p.data.div_(1 + lr*weight_decay)
        

def nag_stepw(optimizer, layer_info, lr, momentum, weight_decay,
             apply_wdecay=True):
    for _, p in layer_info['params']:
        d_p = p.grad.data

        #if weight_decay != 0 and apply_wdecay:
        #    d_p.add_(p.data, alpha=weight_decay)
        
        #if weight_decay != 0 and apply_wdecay and no_prox == True:
        #    p.data.mul_(1-lr*weight_decay)
        
        if momentum != 0:
            param_state = optimizer.state[p]
            if 'momentum_buff' not in param_state:
                param_state['momentum_buff'] = d_p.clone()
            else:
                buf = param_state['momentum_buff']
                buf.mul_(momentum).add_(d_p)
                d_p.add_(buf, alpha=momentum)

        p.data.add_(d_p, alpha=-lr)
        
        if weight_decay != 0 and apply_wdecay:
            p.data.div_(1 + lr*weight_decay)
      

def _adan_step(p, state, lr, beta1=0.9, beta2=0.9, beta3=0.99, eps=1e-8, weight_decay=1e-4):
    grad = p.grad

    if len(state) == 0:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p)
        state['exp_avg_sq'] = torch.zeros_like(p)
        state['exp_avg_diff'] = torch.zeros_like(p)
        state['neg_pre_grad'] = p.grad.clone()
        
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    exp_avg_diff, neg_grad_or_diff = state['exp_avg_diff'], state['neg_pre_grad']
    
    state['step'] += 1
    
    bias_correction1 = 1.0 - beta1 ** state['step']
    bias_correction2 = 1.0 - beta2 ** state['step']
    bias_correction3 = 1.0 - beta3 ** state['step']
    bias_correction3_sqrt = math.sqrt(bias_correction3)
    
    neg_grad_or_diff.add_(grad)
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
    exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff,
                                  alpha=1 - beta2)  # diff_t
    neg_grad_or_diff.mul_(beta2).add_(grad)
    exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff,
                                    neg_grad_or_diff,
                                    value=1 - beta3)  # n_t

    denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1
    
    p.data.addcdiv_(exp_avg, denom, value=-step_size)
    p.data.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
    p.data.div_(1 + lr * weight_decay)

    neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)
    

def adan_step(optimizer, layer_info, lr, betas, eps, weight_decay):
    beta1, beta2, beta3 = betas

    for pname, p in layer_info['params']:
        state = optimizer.state[p]
        if p.grad is None:
            continue

        _adan_step(p, state, lr, beta1, beta2, beta3, eps, weight_decay)