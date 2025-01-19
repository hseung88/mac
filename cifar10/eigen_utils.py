import copy

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh_tridiagonal


def del_attr(obj, names):
    """
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    delete the attribute obj.a.b.c
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    """
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    set the attribute obj.a.b.c to val
    if obj.a.b.c is nn.Parameter, cannot directly use set_attr, need to first use del_attr
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_names_params(model):
    """
    mod: model with nn.Parameters, cannot use functionalized model
    return:
        names_all: a list of all names of mod.paramters, [a1.b1.c1, a2.b2.c2, ...]
        orig_params: tuple of parameters of type nn.Parameter
    """

    orig_params = tuple(model.parameters())
    names_all = []
    for name, p in list(model.named_parameters()):
        names_all.append(name)
    return orig_params, names_all


def vec_to_list(p_vector, model):
    """
    convert a tensor of shape (num_params, ) to a list of tensors
    according to the shape of parameters in a model
    gradient can pass through this operation.
    if param_list is leaf variable, p_vector is not a leaf variable
    """
    p_list = []
    idx = 0
    for param in model.parameters():
        num = param.data.numel()
        a = p_vector[idx:idx + num].clone()
        p_list.append(a.view(param.data.shape))
        idx += num

    return p_list


def list_to_vec(param_list):
    """
    transfer a iterable (can be tuple or list) of tensors to a tensor of shape (num_param, )
    gradient can pass through this operation.
    if param_list is leaf variable, p_vector is not a leaf variable
    """
    p_vector = None
    for p in param_list:
        if p_vector is None:
            p_vector = p.contiguous().view(-1)
        else:
            p_vector = torch.cat([p_vector, p.contiguous().view(-1)])

    return p_vector


def norm_2_list(param_list):
    """
    param_list: list of parameters of a model
    return: 2-norm of parameters in model
    gradient can pass through this operation
    """
    norm = 0.0
    for param in param_list:
        norm += torch.norm(param, p=2)**2

    return torch.sqrt(norm)


def prod_list(u_list, v_list):
    """
    u_list, v_list: lists of tensors
    return: dot product of u_list and v_list
    """

    prod = 0
    for (u, v) in zip(u_list, v_list):
        prod += torch.sum(u*v)

    return prod


def make_functional(mod, names_all, param_iter):
    """
    names_all: list of all names in mod
    param_iter: iterable of parameters, tensor or nn.Parameter

    load param_iter into mod, preserve the type of param_iter
    """
    for name, p in zip(names_all, param_iter):
        del_attr(mod, name.split("."))
        set_attr(mod, name.split("."), p)


def load_weights(mod, names_all, params):
    for name, p in zip(names_all, params):
        set_attr(mod, name.split("."), p)


def functional2(model, data, targets, criterion, device, param_tuple, names_all):
    """
    funCtional of loss w.r.t. param_tuple
    after calling this function, the model becomes functional, does not contain nn.Parameter
    """
    data, targets = data.to(device), targets.to(device)

    make_functional(model, names_all, param_tuple)
    output = model(data)
    loss = criterion(output, targets)

    return loss


def vhp(model, loader, w0_tuple, v_tuple, criterion, device, half=False):
    """
    model: model with nn.Parameter
    w0_tuple: tuple of weights at which hessian is calculated
    v_tuple: tuple of weights of direction
    if half = Ture, then using half type vectors to calculate the Hv. w0_tuple, v_tuple should have type half. Model, loader don't need to be in half in the input.

    return: loss(w0), Hess(w0)*v in format of tuple
            the returned value and u are all in float32 whether half=True or not
    after calling this function, model is still in the format of nn.Parameter
    """
    model = model.to(device)
    criterion = criterion.to(device)

    if half:
        model = model.half()

    value_list, u_list = [], []
    param_tuple_ori, names_all = get_names_params(model)

    for (data, targets) in loader:
        if half:
            data = data.half()

        def f(*param_tuple):
            loss = functional2(model, data, targets, criterion, device, param_tuple, names_all)
            return loss

        value, u = torch.autograd.functional.vhp(f, w0_tuple, v_tuple)
        value_list.append(value)
        u_list.append(list_to_vec(list(u)))

    print("vhp iteration", value, norm_2_list(u))

    # load param_tuple_ori back to model, with type nn.Parameters
    load_weights(model, names_all, param_tuple_ori)

    value, u = sum(value_list) / len(value_list), sum(u_list) / len(u_list)
    value = value.float()
    u = u.float()

    u = tuple(vec_to_list(u, model))

    return value, u


def hess_scipy(model, k, loader, criterion, device):
    """
    calculate eigen values of hessian using scipy.sparse.eigsh
    need to first import eigsh from scipy.sparse
    model: trained model with parameters at minima
    k: number of eigen values to calculate
    loader: data used to calculate hessian
    return: torch.tensor float of eigen values and vectors in cpu, both are sorted in descending order.
    """

    model = model.to(device)
    criterion = criterion.to(device)
    param_list = [_.data for _ in model.parameters()]
    num_params = sum(param.numel() for param in model.parameters())

    def fnc_LO(q):
        """
        q: numpy array of shape (num_params, )
        return: numpy array of shape (num_params, ), calculates Hq, H is the hessian.
        """
        q_list = vec_to_list(torch.tensor(q).float().to(device), model)
        _, v_tuple = vhp(model, loader, tuple(param_list), tuple(q_list), criterion, device)
        v = list_to_vec(v_tuple)

        return v.cpu().detach().numpy()

    A = LinearOperator((num_params, num_params), matvec=fnc_LO)
    eigenvalues, eigenvectors = eigsh(A, k, which='LM')

    idx = list(np.flip(eigenvalues.argsort()))
    eigenvalues = eigenvalues[idx]
    eigenvectors = ((eigenvectors.T[idx]).T)

    eigenvalues = torch.tensor(eigenvalues).float()
    eigenvectors = torch.tensor(eigenvectors).float()

    return eigenvalues, eigenvectors


def hess_lanczo(model, k, loader, criterion, device, half=False):
    """
    calculate eigen values of hessian using Lanczo's method
    model: trained model with parameters at minima
    k: number of eigen values to calculate
    loader: data used to calculate hessian
    half: calculates vhp in half version, use input in normal version

    return: w: np.array of top eigen values sorted in descending order.
    """

    model = model.to(device)
    criterion = criterion.to(device)
    beta, alpha = [0], [0]
    param_list = [_.data for _ in model.parameters()]
    direc_list = [torch.randn(_.shape).to(device) for _ in model.parameters()]
    q0 = [torch.zeros(_.shape).to(device) for _ in model.parameters()]
    q1 = [_ / norm_2_list(direc_list) for _ in direc_list]

    if half:
        for p, q in zip(param_list, q1):
            p, q = p.half(), q.half()

    for i in range(k):
        _, v = vhp(model, loader, tuple(param_list), tuple(q1), criterion, device)
        v = list(v)
        alpha.append(prod_list(q1, v))

        for (ele_v, ele_q0, ele_q1) in zip(v, q0, q1):
            ele_v.data = ele_v.data - beta[i]*ele_q0.data - alpha[i+1]*ele_q1.data

        beta.append(norm_2_list(v))
        q0 = copy.deepcopy(q1)
        w = [ele_v / beta[i+1] for ele_v in v]
        q1 = copy.deepcopy(w)

    # alpha, beta are diagonals of the tridiagonal matrix
    a = np.array([ele.item() for ele in alpha[1:]])
    b = np.array([ele.item() for ele in beta[1:]])
    w, _ = eigh_tridiagonal(a, b[:-1])
    w = -np.sort(-w)

    return w
