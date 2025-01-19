import torch
from torch.optim import Optimizer
from torch.func import vmap
# from torch.profiler import profile, record_function

class SketchySGD(Optimizer):
    """Implements SketchySGD. We assume that there is only one parameter group to optimize.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rank (int): sketch rank
        rho (float): regularization
        lr (float): learning rate
        weight_decay (float): weight decay parameter
        momentum (float): momentum parameter
        chunk_size (int): number of Hessian-vector products to compute in parallel
        verbose (bool): option to print out eigenvalues of Hessian approximation
    """
    def __init__(self, params, rank=10, rho=0.1, lr=0.001, weight_decay=0.0,
                 momentum=0.0, chunk_size=1, verbose=False):
        defaults = dict(rank=rank, rho=rho, lr=lr, weight_decay=weight_decay,
                        momentum=momentum, chunk_size=chunk_size, verbose=verbose)
        self.rank = rank
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.U = None
        self.S = None
        self.momentum = momentum
        self.momentum_buffer = None
        self.n_iters = 0
        super(SketchySGD, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("SketchySGD doesn't currently support per-parameter options (parameter groups)")

    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g = torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params'] if p.grad is not None])
        g = g.detach()

        # update momentum buffer
        if self.momentum_buffer is None: 
            self.momentum_buffer = g
        else:
            self.momentum_buffer = self.momentum * self.momentum_buffer + g

        # one step update
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            rho = group['rho']

            # calculate the preconditioned search direction
            UTg = torch.mv(self.U.t(), self.momentum_buffer / (1 - self.momentum ** (self.n_iters + 1)))
            dir = torch.mv(self.U, (self.S + rho).reciprocal() * UTg) + (self.momentum_buffer / (1 - self.momentum ** (self.n_iters + 1))) / rho - torch.mv(self.U, UTg) / rho

            # update model parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = dir[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-(dp + weight_decay * p.data), alpha=lr)

        self.n_iters += 1

        return loss
    
    # def profile_update_preconditioner(self, grad_tuple):
    #     # Use the profiler context manager to profile CPU and CUDA activities.
    #     with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         # Use the record_function context manager to label blocks of operations.
    #         with record_function("update_preconditioner"):
    #             self.update_preconditioner(grad_tuple)

    #     # After exiting the profiler context, explicitly export the trace to a file.
    #     prof.export_chrome_trace("./logdir/trace.json")

    #     # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #     # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    def update_preconditioner(self, grad_tuple):
        params = []

        for group in self.param_groups:
            for param in group['params']:
                params.append(param)

        gradsH = torch.cat([gradient.view(-1) for gradient in grad_tuple if gradient is not None])

        p = gradsH.shape[0]
        # Generate test matrix (NOTE: This is transposed test matrix)
        Phi = torch.randn((self.rank, p), device = params[0].device) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode = 'reduced')[0].t()

        Y = self._hvp_vmap(gradsH, params)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi
        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())
        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            shift = shift + torch.abs(torch.min(eigs))
            # add shift to eigenvalues
            eigs = eigs + shift
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T 
            C = torch.linalg.cholesky(torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))

        try: 
            B = torch.linalg.solve_triangular(C, Y_shifted, upper = False, left = True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except: 
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to('cpu'), upper = False, left = True).to(C.device)
        _, S, UT = torch.linalg.svd(B, full_matrices = False) # B = V * S * U^T b/c we have been using transposed sketch
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        if self.verbose: 
            print(f'Approximate eigenvalues = {self.S}')
    
    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims = 0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs = v,
                                retain_graph = True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])