from typing import Optional

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing


def friedman_fct(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Friedman function.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (N, 5), where N is the number of samples.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (N,), containing the calculated values of the Friedman function.
    """
    return (1 / 6) * (
        10 * torch.sin(np.pi * X[:, 0] * X[:, 1])  # Term 1
        + 20 * (X[:, 2] - 0.5)  # Term 2
        + 10 * X[:, 3]  # Term 3
        + 5 * X[:, 4]  # Term 4
    )


def positions_to_pdistances(
    positions: torch.Tensor,
    num_neighbors: Optional[int] = None,
    num_dimensions: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculate the pairwise distances between positions.

    Parameters
    ----------
    positions : torch.Tensor
        Holds the positions of the nodes. If num_neighbors and num_dimensions are not None, then the
        first dimension is assumed to represent the batch and the second dimension is assumed to be
        the flattened representation of the positions of the neighbors of each node. If
        num_neighbors and num_dimensions are None, then `positions` is assumed to be an ordinary
        2d tensor where the rows encode the nodes, and the columns encode the dimensions of the
        position of each node.

    num_neighbors : int, optional
        Number of neighbors of each node.

    num_dimensions : int, optional
        Number of dimensions of the positions.

    Returns
    -------
    torch.Tensor
        Tensor containing the pairwise distances between positions. If the input was given in a
        batch format, then the output will be of shape B x N x N, where B is the batch size and N
        is the number of nodes in each graph. If the input was given in a non-batch format, then
        the output will be of shape N x N.
    """
    # Check arguments
    if (num_neighbors is None) != (num_dimensions is None):
        raise ValueError("Either both or neither of num_neighbors and num_dimensions must be None")

    # This if-else ensures positions has shape B x N x D
    if num_neighbors is None:
        positions = positions.unsqueeze(0)
    else:
        positions = positions.reshape(-1, num_neighbors, num_dimensions)

    num_neighbors = positions.shape[1]
    positions_expanded_1 = positions.unsqueeze(2)  # Shape: B x N x 1 x D
    positions_expanded_2 = positions.unsqueeze(1)  # Shape: B x 1 x N x D
    differences = positions_expanded_1 - positions_expanded_2  # Shape: B x N x N x D
    square_diffs = differences.pow(2)
    pairwise_distances = square_diffs.sum(dim=3).sqrt()  # Shape: B x N x N
    return torch.squeeze(pairwise_distances, dim=0)


def distance_to_covariance(
    distances: torch.Tensor, sigma: int | torch.nn.Parameter, phi: int | torch.nn.Parameter
) -> torch.Tensor:
    """
    Calculates the covariance matrix from distances using the squared exponential kernel.

    Parameters
    ----------
    distances : torch.Tensor
        Tensor containing the pairwise distances between points.
    sigma : float
        Scaling factor for the covariance matrix.
    phi : float
        Length scale parameter for the covariance matrix.

    Returns
    -------
    torch.Tensor
        Covariance matrix computed from the distances.

    Notes
    -----
    The returned covariances will have the same shape as the input distances.
    """
    return (sigma**2) * torch.exp(-distances / phi)


def get_covariance_matrices(
    positions: torch.Tensor,
    sigma: int | torch.nn.Parameter,
    phi: int | torch.nn.Parameter,
    tau: int | torch.nn.Parameter,
    num_neighbors: Optional[int] = None,
    num_dimensions: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculates the covariance matrices based on the given positions and parameters.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor containing the positions of the data points.
    sigma : int | torch.nn.Parameter
        The sigma parameter.
    phi : int | torch.nn.Parameter
        The phi parameter.
    tau : int | torch.nn.Parameter
        The tau parameter.
    num_neighbors : Optional[int], optional
        The number of nearest neighbors to consider. Defaults to None.
    num_dimensions : Optional[int], optional
        The number of dimensions of the data points. Defaults to None.

    Returns
    -------
    torch.Tensor
        Tensor containing the covariance matrices.

    Notes
    -----
    If `positions` has a leading batch dimension, the covariance matrices are calculated
    independently for each batch element.

    """
    pairwise_distances = positions_to_pdistances(positions, num_neighbors, num_dimensions)
    covariance_matrices = distance_to_covariance(pairwise_distances, sigma, phi)
    if covariance_matrices.dim() == 2:
        nugget_effect = (tau**2) * torch.eye(covariance_matrices.shape[1])
    else:  # there is a batch dimension
        batch_size = covariance_matrices.shape[0]
        nugget_effect = ((tau**2) * torch.eye(num_neighbors)).repeat(batch_size, 1, 1)
    final_cov_matrices = covariance_matrices + nugget_effect
    return final_cov_matrices


def generate_samples(
    num_samples: int, num_dimensions: int, num_neighbors: int, tau: int, sigma: int, phi: int
) -> torch_geometric.data.Data:
    """
    Generate samples for graph-based learning.

    Parameters
    ----------
    num_samples : int
        The number of samples to generate.
    num_dimensions : int
        The number of positional dimensions for each sample.
    num_neighbors : int
        The number of nearest neighbors to consider for each point.
    tau : float
        The parameter for covariance matrix calculation.
    sigma : float
        The standard deviation for noise distribution.
    phi : float
        The parameter for covariance matrix calculation.

    Returns
    -------
    torch_geometric.data.Data
        The generated samples with covariates, function values, positions, and edges.
    """

    # Define constants
    NUM_FEATURES = 5

    # Sample the covariates, the positions, the function values and the noise
    X = torch.rand(num_samples, NUM_FEATURES)
    f_X = friedman_fct(X)
    S = torch.rand(num_samples, num_dimensions)
    cov_matrix = get_covariance_matrices(S, sigma, phi, tau)
    noise_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(num_samples), cov_matrix
    )
    epsilons = noise_distribution.sample()
    Y = f_X + epsilons

    # Compute the edges of the graph
    distance_matrix = positions_to_pdistances(S)
    edges = []
    neighbor_idc = []

    # Initialize the edges, the edges are predefined for the first m + 1 points
    for i in range(1, num_neighbors + 1):
        for j in range(i):
            edges.append([j, i])
            neighbor_idc.append(j)

    # Find the m nearest neighbors for each remaining point
    for i in range(num_neighbors + 1, num_samples):
        # Find the m nearest neighbors
        neighbors = torch.argsort(distance_matrix[i, :i])[:num_neighbors]
        for j, neighbor in enumerate(neighbors):
            edges.append([neighbor, i])
            neighbor_idc.append(j)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(neighbor_idc).reshape(-1, 1)  # denotes the index of the neighbor
    data = torch_geometric.data.Data(x=X, y=Y, pos=S, edge_index=edge_index, edge_attr=edge_attr)
    assert data.validate(raise_on_error=True)
    return data


class GatherNeighborPositionsConv(MessagePassing):
    """
    The output of node i will be a tensor of shape (num_neighbors, num_dimensions) where the j-th
    row contains the position of the j-th neighbor of node i. The output will actually be flattened,
    so it will be of form (num_neighbors * num_dimensions,)

    Parameters
    ----------
    num_neighbors : int
        The number of neighbors for each node.
    num_dimensions : int
        The number of dimensions for each position vector.

    """

    def __init__(self, num_neighbors, num_dimensions):
        super().__init__(aggr="sum")
        self.num_neighbors = num_neighbors
        self.num_dimensions = num_dimensions

    def forward(self, pos, edge_index, edge_attr):
        return self.propagate(edge_index, pos=pos, edge_attr=edge_attr)

    def message(self, pos_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.num_neighbors * self.num_dimensions)
        col_idc = edge_attr.flatten() * self.num_dimensions
        row_idc = torch.tensor(range(num_edges))
        msg[
            row_idc.unsqueeze(1), col_idc.unsqueeze(1) + torch.tensor(range(self.num_dimensions))
        ] = pos_j
        return msg


class GatherNeighborOutputsConv(MessagePassing):
    """
    The output of node i will be a tensor of shape (num_neighbors+1,) where the j-th row contains
    the output of the (j+1)-th neighbor of node i. The first row will contain the output of node i.
    Assumes that the outputs are already computed.
    """

    def __init__(self, num_neighbors):
        super().__init__(aggr="sum")
        self.num_neighbors = num_neighbors

    def forward(self, o, edge_index, edge_attr):
        out = self.propagate(edge_index, o=o, edge_attr=edge_attr)
        out[:, 0] += o.squeeze()
        return out

    def message(self, o_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.num_neighbors + 1)
        col_idc = edge_attr.flatten() + 1
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = o_j.squeeze()
        return msg


class GatherNeighborTargetsConv(MessagePassing):
    """
    The output of node i will be a tensor of shape (num_neighbors+1,) where the j-th row contains
    the target of the (j+1)-th neighbor of node i. The first row will contain the target of node i.
    """

    def __init__(self, num_neighbors):
        super().__init__(aggr="sum")
        self.num_neighbors = num_neighbors

    def forward(self, y, edge_index, edge_attr):
        out = self.propagate(edge_index, y=y.reshape(-1, 1), edge_attr=edge_attr)
        out[:, 0] += y
        return out

    def message(self, y_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.num_neighbors + 1)
        col_idc = edge_attr.flatten() + 1
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = y_j.squeeze()
        return msg


class CovarianceVectorConv(MessagePassing):
    """
    The output of node i will be Sigma(i, N(i))
    """

    def __init__(self, num_neighbors, sigma, phi):
        super().__init__(aggr="sum")
        self.num_neighbors = num_neighbors
        self.sigma = sigma
        self.phi = phi

    def forward(self, pos, edge_index, edge_attr):
        return self.propagate(edge_index, pos=pos, edge_attr=edge_attr)

    def message(self, pos_i, pos_j, edge_attr):
        squared_diffs = (pos_i - pos_j) ** 2
        distances = torch.sqrt(torch.sum(squared_diffs, dim=1))  # euclidean distance
        covariances = distance_to_covariance(distances, self.sigma, self.phi)
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.num_neighbors)
        col_idc = edge_attr.flatten()
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = covariances
        return msg


class InverseCovMatrixFromPositions(torch.nn.Module):
    """
    Does the following local computation for each node: the represenation is assumed to be a 2d
    tensor containing the positions of the node's neighbors, but flattened. For each node, the
    output will be the inverse of the covariance matrix of all the neighbors' positions, also
    flattened.
    """

    def __init__(self, num_neighbors, num_dimensions, sigma, phi, tau):
        super(InverseCovMatrixFromPositions, self).__init__()
        self.num_neighbors = num_neighbors
        self.num_dimensions = num_dimensions
        self.sigma = sigma
        self.phi = phi
        self.tau = tau

    def forward(self, pos):
        final_cov_matrices = get_covariance_matrices(
            pos, self.sigma, self.phi, self.tau, self.num_neighbors, self.num_dimensions
        )
        # pairwise_distances = positions_to_pdistances(pos, self.num_neighbors, self.num_dimensions)
        # covariance_matrices = distance_to_covariance(pairwise_distances, self.sigma, self.phi)
        batch_size = pos.shape[0]
        # nugget_effect = ((self.tau**2) * torch.eye(self.num_neighbors)).repeat(batch_size, 1, 1)
        # final_cov_matrices = covariance_matrices + nugget_effect
        inverse_cov_matrices = torch.linalg.inv(final_cov_matrices)
        return inverse_cov_matrices.reshape(batch_size, -1)


class InverseCovMatrixConv(torch.nn.Module):
    def __init__(self, num_neighbors, num_dimensions, sigma, phi, tau):
        super(InverseCovMatrixConv, self).__init__()
        self.num_neighbors = num_neighbors
        self.num_dimensions = num_dimensions
        self.sigma = sigma
        self.phi = phi
        self.tau = tau
        self.gather_neighbor_positions = GatherNeighborPositionsConv(num_neighbors, num_dimensions)
        self.compute_cov_matrices = InverseCovMatrixFromPositions(
            num_neighbors, num_dimensions, sigma, phi, tau
        )

    def forward(self, pos, edge_index, edge_attr):
        neighbor_positions = self.gather_neighbor_positions(pos, edge_index, edge_attr)
        return self.compute_cov_matrices(neighbor_positions)


class NNGLS(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_neighbors: int,
        num_dimensions: int,
        sigma: Optional[int] = None,
        phi: Optional[int] = None,
        tau: Optional[int] = None,
    ):
        super(NNGLS, self).__init__()
        self.num_features = num_features
        self.num_neighbors = num_neighbors
        self.num_dimensions = num_dimensions
        self.sigma = torch.nn.Parameter(
            torch.rand(1) * 2 if sigma is None else torch.tensor([sigma])
        )
        self.phi = torch.nn.Parameter(
            torch.rand(1) * 10 + 5 if phi is None else torch.tensor([phi])
        )
        self.tau = torch.nn.Parameter(torch.rand(1) * 2 if tau is None else torch.tensor([tau]))
        self.compute_covariance_vectors = CovarianceVectorConv(num_neighbors, self.sigma, self.phi)
        self.compute_inverse_cov_matrices = InverseCovMatrixFromPositions(
            num_neighbors, num_dimensions, self.sigma, self.phi, self.tau
        )
        self.gather_neighbor_positions = GatherNeighborPositionsConv(num_neighbors, num_dimensions)
        self.gather_neighbor_outputs = GatherNeighborOutputsConv(num_neighbors)
        self.gather_neighbor_targets = GatherNeighborTargetsConv(num_neighbors)

        # Simple MLP to map features to scalars
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )

    def forward(self, batch):
        """
        Forward pass of the NNGLS model.

        Parameters
        ----------
        batch: pytorch_geometric.data.Data
            The input batch containing node features and graph structure.

        Returns
        -------
        tuple
            A tuple containing the decorrelated predictions and decorrelated targets.

        Notes
        -----
        This method performs the forward pass of the model. It takes an input batch, which contains
        node features and graph structure, and computes the decorrelated predictions and
        decorrelated targets.

        The forward pass involves several steps:
        1. Compute covariance vectors Sigma(i,N(i)) using the input batch's node positions, edge
            indices, and edge attributes.
        2. Compute inverse covariance matrices Sigma(N(i),N(i))^-1 by gathering neighbor positions
            and computing the inverse covariance matrices.
        3. Compute the B vectors, i.e., B(i,N(i)), by multiplying the covariance vectors with the
            inverse covariance matrices.
        4. Compute the F scalars, i.e., F(i,i), by subtracting the dot product of B vectors and
            covariance vectors from the sum of sigma^2 and tau^2.
        5. Compute the v vectors, i.e., v(i,N(i)), by dividing the negative extended B vectors by
            the square root of F scalars.
        6. Compute decorrelated predictions by multiplying v vectors with neighbor outputs and
            summing along the second dimension.
        7. Compute decorrelated targets by multiplying v vectors with neighbor targets and summing
            along the second dimension.

        The method returns a tuple containing the decorrelated predictions and decorrelated targets.
        """

        # Compute covariance vectors Sigma(i,N(i))
        cov_vectors = self.compute_covariance_vectors(batch.pos, batch.edge_index, batch.edge_attr)

        # Compute inverse covariance matrices Sigma(N(i),N(i))^-1
        neighbor_positions = self.gather_neighbor_positions(
            batch.pos, batch.edge_index, batch.edge_attr
        )
        inverse_cov_matrices = self.compute_inverse_cov_matrices(neighbor_positions)
        inverse_cov_matrices = inverse_cov_matrices.view(-1, self.num_neighbors, self.num_neighbors)

        # Compute the B vectors, i.e., B(i,N(i))
        b_vectors = torch.matmul(cov_vectors.unsqueeze(1), inverse_cov_matrices).squeeze(1)

        # Compute the F scalars, i.e., F(i,i)
        f_scalars = (self.sigma**2 + self.tau**2) - torch.sum(b_vectors * cov_vectors, dim=1)

        # Compute the v vectors, i.e., v(i,N(i))
        ones = torch.ones(b_vectors.size(0), 1)
        negative_extended_b_vectors = torch.cat((ones, -b_vectors), dim=1)
        v_vectors = negative_extended_b_vectors / torch.sqrt(f_scalars.unsqueeze(1))

        # Compute decorrelated predictions
        batch.o = self.mlp(batch.x)
        neighbor_outputs = self.gather_neighbor_outputs(batch.o, batch.edge_index, batch.edge_attr)
        decorrelated_preds = (v_vectors * neighbor_outputs).sum(dim=1)

        # Compute decorrelated targets
        neighbor_targets = self.gather_neighbor_targets(batch.y, batch.edge_index, batch.edge_attr)
        decorrelated_targets = (v_vectors * neighbor_targets).sum(dim=1)

        # Compute predictions
        preds = torch.sqrt(f_scalars) * (
            decorrelated_preds + torch.sum(b_vectors * neighbor_targets[:, 1:], dim=1)
        )

        return decorrelated_preds, decorrelated_targets, preds
