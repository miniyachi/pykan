import torch
from torch.optim import Optimizer
from torch.func import vmap
import torch.nn.functional as F

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product

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
    def __init__(self, params, lr=0.001, rho=0.1, delta=1.0, chunk_size=1, verbose=False):
        """
        Initialize the SFN optimizer.

        Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1.0).
        damping (float): Damping factor to ensure positive definiteness (default: 1e-4).
        """
        defaults = dict(lr=lr, rho=rho, delta=delta, chunk_size=chunk_size, verbose=verbose)
        self.delta = delta
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.U = None
        self.S = None
        self.n_iters = 0

        super(SFN, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("Currently doesn't support per-parameter options (parameter groups)")

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
        closure (callable, optional): A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        g = torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params'] if p.grad is not None])
        g = g.detach()

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']

            # Compute preconditioned direction
            UTg = torch.mv(self.U.t(), g)
            dir = torch.mv(self.U, (torch.abs(self.S) + rho).reciprocal() * UTg) + g / rho - torch.mv(self.U, UTg) / rho

            # Update model parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = dir[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-dp , alpha=lr)

        self.n_iters += 1
                
        return loss
    

    def update_preconditioner(self, grad_tuple):
        params = []

        for group in self.param_groups:
            for param in group['params']:
                params.append(param)

        gradsH = torch.cat([gradient.view(-1) for gradient in grad_tuple if gradient is not None])

        p = gradsH.shape[0]
        # Generate test matrix (NOTE: This is transposed test matrix)
        crank = int(1.5 * self.rank) # oversampling for truncation 
        Phi = torch.randn((crank, p), device = params[0].device) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode = 'reduced')[0].t()

        # Compute the sketch
        Y = self._hvp_vmap(gradsH, params)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        # Y_shifted = Y + shift * Phi

        # Calculate core matrix Phi^T * H * Phi for eigen-decomposition
        W = torch.mm(Y, Phi.t())
        # W_shifted = torch.mm(Y_shifted, Phi.t())

        # Compute eigen-decomposition of core matrix
        eigs, eigvectors = torch.linalg.eigh(W)
        # eigs, eigvectors = torch.linalg.eigh(W_shifted)
        # eigs = eigs - shift

        # Sort the indices based on the absolute eigvals in descending order
        sorted_indices = torch.argsort(torch.abs(eigs), descending=True)

        # Truncate the eigvals and eigvectors
        sorted_indices = sorted_indices[:self.rank]
        eigs = eigs[sorted_indices]
        eigvectors = eigvectors[:, sorted_indices]

        # Compute pseudo-inverse of the best rank
        W_inv = torch.mm(eigvectors, torch.mm(torch.diag(1.0 / eigs), eigvectors.t()))

        # Compute eigen-decomposition of Nystrom approximation
        Q, R = torch.linalg.qr(Y.t(), mode = 'reduced')
        eigtarget = torch.mm(R, torch.mm(W_inv, R.t()))
        self.S, eigvectors = torch.linalg.eigh(eigtarget)
        self.U = torch.mm(Q, eigvectors)

        if self.verbose: 
            print(f'Approximate eigenvalues = {self.S}')
    
    def appx_min_eigvec(self, grad_tuple):
        return None
    
    def _fvp(self, grad_params, params, v): # v = [v, t]
        # Compute Hessian-vector product
        Hv = self._hvp(grad_params, params, v[:-1])

        # Multiply the last element of v with grad_params
        tg = v[-1] * grad_params

        # Compute gTv
        gTv = torch.dot(grad_params, v[:-1])

        Hv_tg = Hv + tg
        scalar = (gTv - self.delta * v[-1]).view(1)
        
        return torch.cat([Hv_tg, scalar])


    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims = 0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs = v,
                                retain_graph = True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

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
