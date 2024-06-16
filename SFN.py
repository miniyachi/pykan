import torch
from torch.optim import Optimizer
from torch.func import vmap
import torch.nn.functional as F

from pyhessian.utils import group_product, group_add, orthnormal, normalization
from opt_utils import group_scalar

class SFN(Optimizer):
    """
    Implements Saddle Free Newton. We assume that there is only one parameter group to optimize.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rank (int): sketch rank
        rho (float): regularization
        lr (float): learning rate
        weight_decay (float): weight decay parameter
        chunk_size (int): number of Hessian-vector products to compute in parallel
        verbose (bool): option to print out eigenvalues of Hessian approximation
    """
    def __init__(self, params, lr=0.001, radius=0.1, delta=0.01, chunk_size=1, verbose=False):
        """
        Initialize the SFN optimizer.

        Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1.0).
        damping (float): Damping factor to ensure positive definiteness (default: 1e-4).
        """
        defaults = dict(lr=lr, radius=radius, delta=delta, chunk_size=chunk_size, verbose=verbose)
        self.delta = delta
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.n_iters = 0

        super(SFN, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("Currently doesn't support per-parameter options (parameter groups)")

    def step(self, gradsH_tuple, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Get the parameters as a list
        params = []
        # gradsH = []
        for group in self.param_groups:
            for param in group['params']:
                params.append(param)
                # if param.grad is not None:
                #     gradsH.append(param.grad)
        
        # Get the gradients as a list
        gradsH = []
        for gradient in gradsH_tuple:
            if gradient is not None:
                gradsH.append(gradient)

        # Compute the augmented direction
        _, dir_aug = self.appx_min_eigvec(gradsH, params)

        # Obtain homogenized direction
        if torch.abs(dir_aug[-1]) > 1e-6:
            dir = group_scalar(dir_aug[:-1], 1/dir_aug[-1])
        else:
            dir = group_scalar(dir_aug[:-1], torch.sign(-group_product(gradsH, dir_aug[:-1])))

        # Check if d is a descent direction
        if group_product(dir, gradsH) >= 0:
            print("Warning: dir is not a descent direction. dot(grad, d) = ", group_product(dir, gradsH))

        for group in self.param_groups:
            lr = group['lr']

        #     norm_dir = torch.sqrt(sum([torch.norm(d)**2 for d in dir]))
        #     if norm_dir > lr:
        #         alpha = lr / norm_dir
        #         alpha = alpha.cpu().item()
        #         if self.verbose:
        #             print(f'Projected direction with norm {norm_dir} to norm {lr} with alpha = {alpha}')
        #     else:
        #         alpha = 1.0
        #         if self.verbose:
        #             print(f'Projected direction with norm {norm_dir} without scaling')

            # Update model parameters
            for (p, dp) in zip(group['params'], dir):
                p.data.add_(dp , alpha=lr)

        self.n_iters += 1
                
        return loss
    
    
    def appx_min_eigvec(self, gradsH, params, iter=100):

        device = params[0].device

        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
        # augment the vector with a scalar
        v.append(torch.randint(2, (1,), device=device))
        v = normalization(v)

        # standard Lanczos algorithm initialization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        ############### Lanczos
        for i in range(iter):
            self.zero_grad()
            Fv = [torch.zeros(p.size()).to(device) for p in params]
            Fv.append(torch.zeros(1).to(device))   # add a scalar
            if i == 0:
                Fv = self._fvp(gradsH, params, v)
                alpha = group_product(Fv, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(Fv, v, alpha=-alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).to(device) for p in params]
                    w.append(torch.randn(1).to(device))
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                Fv = self._fvp(gradsH, params, v)
                alpha = group_product(Fv, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(Fv, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)

        T = torch.zeros(iter, iter).to(device)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        eigvals, eigvecs_T = torch.linalg.eigh(T)
        
        V = torch.stack([torch.cat([v_i.reshape(-1) for v_i in v]) for v in v_list])
        tmp_vec = torch.mv(V.t(), eigvecs_T[:, 0])

        ls = 0
        min_eigvec = []
        for p in params:
                np = torch.numel(p)
                min_eigvec.append(tmp_vec[ls:ls+np].view(p.shape))
                ls += np
        min_eigvec.append(tmp_vec[ls:])

        if self.verbose:             
            # Compute the residual
            Fv = self._fvp(gradsH, params, min_eigvec)
            lambdav = group_scalar(min_eigvec, eigvals[0])
            r = group_add(Fv, lambdav, alpha=-1.0)
            res = group_product(r, r)**0.5

            print(f'Approximate minimum eigenvalue = {eigvals[0]},  Eigenvector residual = {res}')
        
        return eigvals[0], min_eigvec
    
    def _fvp(self, grad_params, params, v): # v = [u, t]
        # Compute Hessian-vector product
        Hv = self._hvp(grad_params, params, v[:-1])

        # Multiply the last element of v with grad_params
        tg = group_scalar(grad_params, v[-1])

        # Compute gTv
        gTv = group_product(grad_params, v[:-1])

        output = group_add(Hv, tg)
        output.append(gTv - self.delta * v[-1])
        output = [out_i.detach() for out_i in output]
        
        return output


    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp_cat(grad_params, params, v), in_dims = 0, chunk_size=self.chunk_size)

    def _hvp_cat(self, grad_params, params, v):
        return torch.cat([Hvi.reshape(-1) for Hvi in self._hvp(grad_params, params, v)])

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs = v,
                                 retain_graph = True)
        return [Hvi.detach() for Hvi in Hv]

# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    optimizer = SFN(model.parameters(), lr=1.0, damping=1e-4)

    # Dummy data
    input = torch.randn(10)
    target = torch.randn(1)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()  # Clear the gradients
        output = model(input)  # Forward pass
        loss = F.mse_loss(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
